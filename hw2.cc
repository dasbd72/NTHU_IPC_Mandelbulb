#include <lodepng.h>
#include <mpi.h>
#include <omp.h>

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstdlib>

#define GLM_FORCE_SWIZZLE  // vec3.xyz(), vec3.xyx() ...ect, these are called "Swizzle".
// https://glm.g-truc.net/0.9.1/api/a00002.html
//
#include <glm/glm.hpp>
// for the usage of glm functions
// please refer to the document: http://glm.g-truc.net/0.9.9/api/a00143.html
// or you can search on google with typing "glsl xxx"
// xxx is function name (eg. glsl clamp, glsl smoothstep)

#define pi 3.1415926535897932384626433832795

typedef glm::dvec2 vec2;  // doube precision 2D vector (x, y) or (u, v)
typedef glm::dvec3 vec3;  // 3D vector (x, y, z) or (r, g, b)
typedef glm::dvec4 vec4;  // 4D vector (x, y, z, w)
typedef glm::dmat3 mat3;  // 3x3 matrix

unsigned int num_threads;  // number of thread
unsigned int width;        // image width
unsigned int height;       // image height
vec2 iResolution;          // just for convenience of calculation

const int AA = 2;  // anti-aliasing
const int SQAA = AA * AA;

const double power = 8.0;           // the power of the mandelbulb equation
const double md_iter = 24;          // the iteration count of the mandelbulb
const double ray_step = 10000;      // maximum step of ray marching
const double shadow_step = 1500;    // maximum step of shadow casting
const double step_limiter = 0.2;    // the limit of each step length
const double ray_multiplier = 0.1;  // prevent over-shooting, lower value for higher quality
const double bailout = 2.0;         // escape radius
const double eps = 0.0005;          // precision
const double FOV = 1.5;             // fov ~66deg
const double far_plane = 100.;      // scene depth

vec3 camera_pos;  // camera position in 3D space (x, y, z)
vec3 target_pos;  // target position in 3D space (x, y, z)

unsigned char* raw_image;  // 1D image
unsigned char** image;     // 2D image

int world_rank;  // Mpi world_rank
int world_size;  // Mpi world_size
double* raw_local_color;
double** local_color;
double* raw_global_color;
double** global_color;
// unsigned int* tasks;
unsigned int total_tasks;
clock_t start_time, end_time;

// save raw_image to PNG file
void write_png(const char* filename) {
    unsigned error = lodepng_encode32_file(filename, raw_image, width, height);

    if (error)
        printf("png error %u: %s\n", error, lodepng_error_text(error));
}

// mandelbulb distance function (DE)
// v = v^8 + c
// p: current position
// trap: for orbit trap coloring : https://en.wikipedia.org/wiki/Orbit_trap
// return: minimum distance to the mandelbulb surface
double md(vec3 p, double& trap) {
    vec3 v = p;
    double dr = 1.;             // |v'|
    double r = glm::length(v);  // r = |v| = sqrt(x^2 + y^2 + z^2)
    trap = r;

    for (int i = 0; i < md_iter; ++i) {
        double theta = glm::atan(v.y, v.x) * power;
        double phi = glm::asin(v.z / r) * power;
        dr = power * glm::pow(r, power - 1.) * dr + 1.;
        v = p + glm::pow(r, power) *
                    vec3(cos(theta) * cos(phi), cos(phi) * sin(theta), -sin(phi));  // update vk+1

        // orbit trap for coloring
        trap = glm::min(trap, r);

        r = glm::length(v);  // update r
        if (r > bailout)
            break;  // if escaped
    }
    return 0.5 * log(r) * r / dr;  // mandelbulb's DE function
}

// scene mapping
double map(vec3 p, double& trap, int& ID) {
    const vec2 rt = vec2(cos(pi / 2.), sin(pi / 2.));
    vec3 rp = mat3(1., 0., 0., 0., rt.x, -rt.y, 0., rt.y, rt.x) *
              p;  // rotation matrix, rotate 90 deg (pi/2) along the X-axis
    ID = 1;
    return md(rp, trap);
}

// dummy function
// becase we dont need to know the ordit trap or the object ID when we are calculating the surface
// normal
double map(vec3 p) {
    double dmy;  // dummy
    int dmy2;    // dummy2
    return map(p, dmy, dmy2);
}

// simple palette function (borrowed from Inigo Quilez)
// see: https://www.shadertoy.com/view/ll2GD3
vec3 pal(double t, vec3 a, vec3 b, vec3 c, vec3 d) {
    return a + b * glm::cos(2. * pi * (c * t + d));
}

// second march: cast shadow
// also borrowed from Inigo Quilez
// see: http://www.iquilezles.org/www/articles/rmshadows/rmshadows.htm
double softshadow(vec3 ro, vec3 rd, double k) {
    double res = 1.0;
    double t = 0.;  // total distance
    for (int i = 0; i < shadow_step; ++i) {
        double h = map(ro + rd * t);
        res = glm::min(
            res, k * h / t);  // closer to the objects, k*h/t terms will produce darker shadow
        if (res < 0.02)
            return 0.02;
        t += glm::clamp(h, .001, step_limiter);  // move ray
    }
    return glm::clamp(res, .02, 1.);
}

// use gradient to calc surface normal
vec3 calcNor(vec3 p) {
    vec2 e = vec2(eps, 0.);
    return normalize(vec3(map(p + e.xyy()) - map(p - e.xyy()),  // dx
                          map(p + e.yxy()) - map(p - e.yxy()),  // dy
                          map(p + e.yyx()) - map(p - e.yyx())   // dz
                          ));
}

// first march: find object's surface
double trace(vec3 ro, vec3 rd, double& trap, int& ID) {
    double t = 0;    // total distance
    double len = 0;  // current distance

    for (int i = 0; i < ray_step; ++i) {
        len = map(ro + rd * t, trap,
                  ID);  // get minimum distance from current ray position to the object's surface
        if (glm::abs(len) < eps || t > far_plane)
            break;
        t += len * ray_multiplier;
    }
    return t < far_plane
               ? t
               : -1.;  // if exceeds the far plane then return -1 which means the ray missed a shot
}

int main(int argc, char** argv) {
    // ./source [num_threads] [x1] [y1] [z1] [x2] [y2] [z2] [width] [height] [filename]
    // num_threads: number of threads per process
    // x1 y1 z1: camera position in 3D space
    // x2 y2 z2: target position in 3D space
    // width height: image size
    // filename: filename
    assert(argc == 11);

    //---init arguments
    num_threads = atoi(argv[1]);
    camera_pos = vec3(atof(argv[2]), atof(argv[3]), atof(argv[4]));
    target_pos = vec3(atof(argv[5]), atof(argv[6]), atof(argv[7]));
    width = atoi(argv[8]);
    height = atoi(argv[9]);
    total_tasks = width * height * SQAA;
    iResolution = vec2(width, height);

    // tasks = new unsigned int[total_tasks];
    // for (int pix = 0; pix < total_tasks; ++pix) tasks[pix] = pix;
    // std::random_shuffle(tasks, tasks + total_tasks);
    //---

    //===MPI tasks=====================================================================================
    // start_time = clock();
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    //---create image
    raw_image = new unsigned char[width * height * SQAA];
    image = new unsigned char*[height];
#pragma omp parallel for schedule(dynamic) num_threads(num_threads)
    for (int i = 0; i < height; ++i) {
        image[i] = raw_image + i * width * SQAA;
    }
    //---

    //---create local color
    raw_local_color = new double[width * height * (SQAA - 1)];
    memset(raw_local_color, 0, width * height * (SQAA - 1) * sizeof(double));
    local_color = new double*[height];
#pragma omp parallel for schedule(dynamic) num_threads(num_threads)
    for (int i = 0; i < height; ++i) {
        local_color[i] = raw_local_color + i * width * (SQAA - 1);
    }
    //---

    //---create global color
    raw_global_color = new double[width * height * (SQAA - 1)];
    global_color = new double*[height];
#pragma omp parallel for schedule(dynamic) num_threads(num_threads)
    for (int i = 0; i < height; ++i) {
        global_color[i] = raw_global_color + i * width * (SQAA - 1);
    }
    //---

    //---start rendering
#pragma omp parallel for schedule(dynamic) num_threads(num_threads)
    for (int iter = world_rank; iter < total_tasks; iter += world_size) {
        int i = (iter >> 2) / width;
        int j = (iter >> 2) % width;
        int n = (iter & (1 << 1)) ? 1 : 0;
        int m = (iter & 1) ? 1 : 0;

        vec2 p = vec2(j, i) + vec2(m, n) / (double)AA;

        vec2 uv = (-iResolution.xy() + 2. * p) / iResolution.y;
        uv.y *= -1;  // flip upside down
        //---

        //---create camera
        vec3 cf = glm::normalize(target_pos - camera_pos);  // forward vector
        vec3 cs =
            glm::normalize(glm::cross(cf, vec3(0., 1., 0.)));        // right (side) vector
        vec3 cu = glm::normalize(glm::cross(cs, cf));                // up vector
        vec3 rd = glm::normalize(uv.x * cs + uv.y * cu + FOV * cf);  // ray direction
        //---

        //---marching
        double trap;  // orbit trap
        int objID;    // the object id intersected with
        double d = trace(camera_pos, rd, trap, objID);
        //---

        //---lighting
        vec3 col(0.);                          // color
        vec3 sd = glm::normalize(camera_pos);  // sun direction (directional light)
        vec3 sc = vec3(1., .9, .717);          // light color
        //---

        //---coloring
        if (d < 0.) {        // miss (hit sky)
            col = vec3(0.);  // sky color (black)
        } else {
            vec3 pos = camera_pos + rd * d;      // hit position
            vec3 nr = calcNor(pos);              // get surface normal
            vec3 hal = glm::normalize(sd - rd);  // blinn-phong lighting model (vector
                                                 // h)
            // for more info:
            // https://en.wikipedia.org/wiki/Blinn%E2%80%93Phong_shading_model

            // use orbit trap to get the color
            col = pal(trap - .4, vec3(.5), vec3(.5), vec3(1.),
                      vec3(.0, .1, .2));  // diffuse color
            vec3 ambc = vec3(0.3);        // ambient color
            double gloss = 32.;           // specular gloss

            // simple blinn phong lighting model
            double amb =
                (0.7 + 0.3 * nr.y) *
                (0.2 + 0.8 * glm::clamp(0.05 * log(trap), 0.0, 1.0));  // self occlution
            double sdw = softshadow(pos + .001 * nr, sd, 16.);         // shadow
            double dif = glm::clamp(glm::dot(sd, nr), 0., 1.) * sdw;   // diffuse
            double spe = glm::pow(glm::clamp(glm::dot(nr, hal), 0., 1.), gloss) *
                         dif;  // self shadow

            vec3 lin(0.);
            lin += ambc * (.05 + .95 * amb);  // ambient color * ambient
            lin += sc * dif * 0.8;            // diffuse * light color * light intensity
            col *= lin;

            col = glm::pow(col, vec3(.7, .9, 1.));  // fake SSS (subsurface scattering)
            col += spe * 0.8;                       // specular
        }
        //---

        col = glm::clamp(glm::pow(col, vec3(.4545)), 0., 1.);  // gamma correction

#pragma omp critical
        {
            local_color[i][(SQAA - 1) * j + 0] += col.r;
            local_color[i][(SQAA - 1) * j + 1] += col.g;
            local_color[i][(SQAA - 1) * j + 2] += col.b;
        }
    }
    //---

    MPI_Reduce(raw_local_color, raw_global_color, height * width * (SQAA - 1), MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Finalize();
    // end_time = clock();
    // printf("Rank %d : %lf\n", world_rank, (double)(end_time - start_time) * 1000. / CLOCKS_PER_SEC);
    //===MPI tasks=====================================================================================

    if (world_rank == 0) {
#pragma omp parallel for schedule(dynamic) num_threads(num_threads) collapse(2)
        for (int i = 0; i < height; ++i) {
            for (int j = 0; j < width; ++j) {
                for (int c = 0; c < (SQAA - 1); ++c) {
                    global_color[i][(SQAA - 1) * j + c] /= (double)(SQAA);
                    global_color[i][(SQAA - 1) * j + c] *= 255.0;
                    image[i][SQAA * j + c] = (unsigned char)global_color[i][(SQAA - 1) * j + c];  // rgb
                }
                image[i][SQAA * j + 3] = 255;  // a
            }
        }
    }

    //---saving image
    if (world_rank == 0) {
        write_png(argv[10]);
    }
    //---

    //---finalize
    delete[] raw_image;
    delete[] image;
    delete[] raw_local_color;
    delete[] local_color;
    delete[] raw_global_color;
    delete[] global_color;
    // delete[] tasks;
    //---

    return 0;
}
/*
make;time srun -N2 -n3 -c4 -t0:100 ./hw2 4 2 0.1 0.1 0 0 0 128 128 output/test.png;
make;time srun -N2 -n3 -c4 -t0:100 ./hw2 4 0.1 2 0.1 0 0 0 128 128 output/test.png;
make;time srun -N2 -n3 -c4 -t0:100 ./hw2 4 0.1 0.1 2 0 0 0 128 128 output/test.png;

make;time srun -N2 -n3 -c4 -t0:5 ./hw2 4 -0.522 2.874 1.340 0 0 0 64 64 output/01.png;                              hw2-diff output/01.png testcases/01.png
make;time srun -N2 -n3 -c4 -t0:15 ./hw2 4 4.152 2.398 -2.601 0 0 0 128 128 output/02.png;                           hw2-diff output/02.png testcases/02.png
make;time srun -N2 -n2 -c12 -t0:150 ./hw2 12 1.885 -1.570 3.213 0 0 0 512 512 output/03.png;                        hw2-diff output/03.png testcases/03.png
make;time srun -N3 -n6 -c6 -t0:250 ./hw2 6 -0.027 -0.097 3.044 0 0 0 512 512 output/04.png;                         hw2-diff output/04.png testcases/04.png
make;time srun -N2 -n4 -c6 -t0:150 ./hw2 6 3.726 0.511 -0.096 0 0 0 512 512 output/05.png;                          hw2-diff output/05.png testcases/05.png
make;time srun -N4 -n4 -c12 -t0:180 ./hw2 12 0.7725 -0.385 1.3065 0.782 -0.178 0.312 1024 1024 output/06.png;       hw2-diff output/06.png testcases/06.png
make;time srun -N4 -n4 -c12 -t0:180 ./hw2 12 1.1187 -1.234 -0.285 -0.282 -0.312 -0.378 1024 1024 output/07.png;     hw2-diff output/07.png testcases/07.png
make;time srun -N4 -n4 -c12 -t0:210 ./hw2 12 1.1645 2.0475 1.7305 -0.8492 -1.8767 -1.00928 1536 1536 output/08.png; hw2-diff output/08.png testcases/08.png

make;time srun -N2 -n4 -c12 ./hw2 6 2 2 2 0 0 0 1920 1080 output/test.png
 */