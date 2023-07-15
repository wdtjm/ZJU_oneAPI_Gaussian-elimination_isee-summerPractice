#include <sycl/sycl.hpp>
#include <vector>
#include <queue>
#include <iostream>
#include <string>
#if FPGA || FPGA_EMULATOR
#include <sycl/ext/intel/fpga_extensions.hpp>
#endif
using namespace sycl;
#define N 1024

int* createMatrix(int n) {
    return new int[n * n];
}

void init(int* a1, int* b1, int n) {
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            a1[i * n + j] = 1;
            b1[i * n + j] = 1;
        }
    }
}

void print(int* a1, int n,FILE* f1) {
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            fprintf(f1,"%5d ", a1[i * n + j]);
        }
        fprintf(f1,"\n");
    }
}

void sub(int* a1, int* b1, int* c1, int n, queue q) {
    buffer a2(a1, range<2>{n, n});
    buffer b2(b1, range<2>{n, n});
    buffer c2(c1, range<2>{n, n});
    q.submit([&](handler& h) {
        accessor a(a2, h, read_only);
        accessor b(b2, h, read_only);
        accessor c(c2, h, write_only);
        h.parallel_for(range<2>{n, n}, [=](id<2> i) {
            c[i] = a[i] - b[i];
            });
        });
}

void add(int* a1, int* b1, int* c1, int n, queue q) {
    buffer a2(a1, range<2>{n, n});
    buffer b2(b1, range<2>{n, n});
    buffer c2(c1, range<2>{n, n});
    q.submit([&](handler& h) {
        accessor a(a2, h, read_only);
        accessor b(b2, h, read_only);
        accessor c(c2, h, write_only);
        h.parallel_for(range<2>{n, n}, [=](id<2> i) {
            c[i] = a[i] + b[i];
            });
        });
}

void mul(int* a1, int* b1, int* c1, int n, queue q) {
    buffer a2(a1, range<2>{n, n});
    buffer b2(b1, range<2>{n, n});
    buffer c2(c1, range<2>{n, n});
    q.submit([&](handler& h) {
        accessor a(a2, h, read_only);
        accessor b(b2, h, read_only);
        accessor c(c2, h, write_only);
        h.parallel_for(range<2>{n, n}, [=](id<2> i) {
            int s = 0;
            for (int j = 0; j < n; j++) {
                s += a[i[0]][j] * b[j][i[1]];
            }
            c[i] = s;
            });
        });
}

void Strassen(int* a, int* b, int* c, int n, queue q) {
    int* s1 = createMatrix(n / 2);
    int* s2 = createMatrix(n / 2);
    int* s3 = createMatrix(n / 2);
    int* s4 = createMatrix(n / 2);
    int* s12 = createMatrix(n / 2);
    int* s22 = createMatrix(n / 2);
    int* s32 = createMatrix(n / 2);
    int* s42 = createMatrix(n / 2);
    int* s13 = createMatrix(n / 2);
    int* s23 = createMatrix(n / 2);
    int* s33 = createMatrix(n / 2);
    int* s43 = createMatrix(n / 2);
    for (int i = 0; i < n / 2; i++) {
        for (int j = 0; j < n / 2; j++) {
            s1[i * N / 2 + j] = a[i * N + j];
            s12[i * N / 2 + j] = b[i * N + j];
            s2[i * N / 2 + j] = a[i * N + j + N / 2];
            s22[i * N / 2 + j] = b[i * N + j + N / 2];
            s3[i * N / 2 + j] = a[(i + N / 2) * N + j];
            s32[i * N / 2 + j] = b[(i + N / 2) * N + j];
            s4[i * N / 2 + j] = a[(i + N / 2) * N + j + N / 2];
            s42[i * N / 2 + j] = b[(i + N / 2) * N + j + N / 2];
        }
    }
    int* c1 = createMatrix(n / 2);
    int* c2 = createMatrix(n / 2);
    int* c3 = createMatrix(n / 2);
    int* c4 = createMatrix(n / 2);
    int* c5 = createMatrix(n / 2);
    int* c6 = createMatrix(n / 2);
    int* c7 = createMatrix(n / 2);
    int* c8 = createMatrix(n / 2);
    int* c9 = createMatrix(n / 2);
    int* c10 = createMatrix(n / 2);
    add(s1, s4, c1, N / 2, q);
    add(s12, s42, c2, N / 2, q);
    add(s3, s4, c3, N / 2, q);
    sub(s22, s42, c4, N / 2, q);
    sub(s32, s12, c5, N / 2, q);
    add(s1, s2, c6, N / 2, q);
    sub(s3, s1, c7, N / 2, q);
    add(s12, s22, c8, N / 2, q);
    sub(s2, s4, c9, N / 2, q);
    add(s32, s42, c10, N / 2, q);
    int* m1 = createMatrix(n / 2);
    int* m2 = createMatrix(n / 2);
    int* m3 = createMatrix(n / 2);
    int* m4 = createMatrix(n / 2);
    int* m5 = createMatrix(n / 2);
    int* m6 = createMatrix(n / 2);
    int* m7 = createMatrix(n / 2);
    mul(c1, c2, m1, N / 2, q);
    mul(c3, s12, m2, N / 2, q);
    mul(s1, c4, m3, N / 2, q);
    mul(s4, c5, m4, N / 2, q);
    mul(c6, s42, m5, N / 2, q);
    mul(c7, c8, m6, N / 2, q);
    mul(c9, c10, m7, N / 2, q);
    add(m1, m4, s13, N / 2, q);
    sub(s13, m5, s13, N / 2, q);
    add(s13, m7, s13, N / 2, q);
    add(m3, m5, s23, N / 2, q);
    add(m2, m4, s33, N / 2, q);
    sub(m1, m2, s43, N / 2, q);
    add(s43, m3, s43, N / 2, q);
    add(s43, m6, s43, N / 2, q);
    for (int i = 0; i < n / 2; i++) {
        for (int j = 0; j < n / 2; j++) {
            c[i * N + j] = s13[i * N / 2 + j];
            c[i * N + j + N / 2] = s23[i * N / 2 + j];
            c[(i + N / 2) * N + j] = s33[i * N / 2 + j];
            c[(i + N / 2) * N + j + N / 2] = s43[i * N / 2 + j];
        }
    }
    delete[]s1;
    delete[]s2;
    delete[]s3;
    delete[]s4;
    delete[]s12;
    delete[]s22;
    delete[]s32;
    delete[]s42;
    delete[]s13;
    delete[]s23;
    delete[]s33;
    delete[]s43;
    delete[]c1;
    delete[]c2;
    delete[]c3;
    delete[]c4;
    delete[]c5;
    delete[]c6;
    delete[]c7;
    delete[]c8;
    delete[]c9;
    delete[]c10;
    delete[]m1;
    delete[]m2;
    delete[]m3;
    delete[]m4;
    delete[]m5;
    delete[]m6;
    delete[]m7;
}

int main(int argc, char* argv[]) {
    FILE* f1;
    f1 = fopen("a.out", "w");
    if (f1 == NULL) printf("文件未打开\n");
    else {
        queue q(cpu_selector{});
        int* a = createMatrix(N);
        int* b = createMatrix(N);
        int* c = createMatrix(N);

        init(a, b, N);
        Strassen(a, b, c, N, q);
        print(a, N, f1);
        print(b, N, f1);
        print(c, N, f1);

        delete[] a;
        delete[] b;
        delete[] c;

        return 0;
    }
}