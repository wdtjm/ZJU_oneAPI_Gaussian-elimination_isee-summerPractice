#include <sycl/sycl.hpp>
#include <vector>
#include <queue>
#include <iostream>
#include <string>
#if FPGA || FPGA_EMULATOR
#include <sycl/ext/intel/fpga_extensions.hpp>
#endif
using namespace sycl;
#define N 256

int* createMatrix(int n) { //申请空间创造动态数组
    return new int[n * n];
}

void init(int* a1, int* b1, int n) { //初始化数组，用0-9的随机数进行填充
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            a1[i * n + j] = rand()%10;
            b1[i * n + j] = rand()%10; //实现随机数填充
        }
    }
}

void print(int* a1, int n,FILE* f) { //将数据输出到文件a.out中
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            fprintf(f,"%d  ", a1[i * n + j]);
        }
        fprintf(f,"\n");
    }
    fprintf(f, "\n");
}

void sub(int* a1, int* b1, int* c1, int n, queue q) { //矩阵减法
    buffer a2(a1, range<2>{n, n});
    buffer b2(b1, range<2>{n, n});
    buffer c2(c1, range<2>{n, n}); //创建buffer用于并行化计算
    q.submit([&](handler& h) {
        accessor a(a2, h, read_only);
        accessor b(b2, h, read_only);
        accessor c(c2, h, write_only); //创建访问器用于读写操作
        h.parallel_for(range<2>{n, n}, [=](id<2> i) { //并行操作
            c[i] = a[i] - b[i];
            });
        });
}

void add(int* a1, int* b1, int* c1, int n, queue q) { //矩阵加法
    buffer a2(a1, range<2>{n, n});
    buffer b2(b1, range<2>{n, n});
    buffer c2(c1, range<2>{n, n}); //创建buffer用于并行化计算
    q.submit([&](handler& h) {
        accessor a(a2, h, read_only);
        accessor b(b2, h, read_only);
        accessor c(c2, h, write_only); //创建访问器用于读写操作
        h.parallel_for(range<2>{n, n}, [=](id<2> i) { //并行操作
            c[i] = a[i] + b[i];
            });
        });
}

void mul(int* a1, int* b1, int* c1, int n, queue q) { //并行算法计算矩阵乘法
    buffer a2(a1, range<2>{n, n});
    buffer b2(b1, range<2>{n, n});
    buffer c2(c1, range<2>{n, n}); //创建buffer用于并行化计算
    q.submit([&](handler& h) {
        accessor a(a2, h, read_only);
        accessor b(b2, h, read_only);
        accessor c(c2, h, write_only); //创建访问器用于读写操作
        h.parallel_for(range<2>{n, n}, [=](id<2> i) { //并行操作
            int s = 0;
            for (int j = 0; j < n; j++) { //计算单一格子的数值
                s += a[i[0]][j] * b[j][i[1]];
            }
            c[i] = s;
            });
        });
}
void mul2(int* a1, int* b1, int* c1, int n) { //串行方法计算矩阵乘法
    for (int i = 0; i < n;i++) {
        for (int j = 0; j < n;j++) {
            int s = 0;
            for (int k = 0; k < n;k++) {
                s += a1[i * n + k] * b1[k * n + j];
            }
            c1[i * n + j] = s;
        }
    }
}

void Strassen(int* a, int* b, int* c, int n, queue q) { //算法主函数
    if (n<=128) { //递归出口
        mul2(a, b, c, n);
        return;
    }
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
    int* s43 = createMatrix(n / 2); //创建分块矩阵
    for (int i = 0; i < n / 2; i++) {
        for (int j = 0; j < n / 2; j++) {
            s1[i * n / 2 + j] = a[i * n + j];
            s12[i * n / 2 + j] = b[i * n + j];
            s2[i * n / 2 + j] = a[i * n + j + n / 2];
            s22[i * n / 2 + j] = b[i * n + j + n / 2];
            s3[i * n / 2 + j] = a[(i + n / 2) * n + j];
            s32[i * n / 2 + j] = b[(i + n / 2) * n + j];
            s4[i * n / 2 + j] = a[(i + n / 2) * n + j + n / 2];
            s42[i * n / 2 + j] = b[(i + n / 2) * n + j + n / 2];
        }
    } //分块矩阵填充，实现矩阵分割
    int* c1 = createMatrix(n / 2);
    int* c2 = createMatrix(n / 2);
    int* c3 = createMatrix(n / 2);
    int* c4 = createMatrix(n / 2);
    int* c5 = createMatrix(n / 2);
    int* c6 = createMatrix(n / 2);
    int* c7 = createMatrix(n / 2);
    int* c8 = createMatrix(n / 2);
    int* c9 = createMatrix(n / 2);
    int* c10 = createMatrix(n / 2); //辅助矩阵创建
    add(s1, s4, c1, n / 2, q);
    add(s12, s42, c2, n / 2, q);
    add(s3, s4, c3, n / 2, q);
    sub(s22, s42, c4, n / 2, q);
    sub(s32, s12, c5, n / 2, q);
    add(s1, s2, c6, n / 2, q);
    sub(s3, s1, c7, n / 2, q);
    add(s12, s22, c8, n / 2, q);
    sub(s2, s4, c9, n / 2, q);
    add(s32, s42, c10, n / 2, q); //辅助矩阵填充
    int* m1 = createMatrix(n / 2);
    int* m2 = createMatrix(n / 2);
    int* m3 = createMatrix(n / 2);
    int* m4 = createMatrix(n / 2);
    int* m5 = createMatrix(n / 2);
    int* m6 = createMatrix(n / 2);
    int* m7 = createMatrix(n / 2); //计算矩阵创建
    Strassen(c1, c2, m1, n / 2, q);
    Strassen(c3, s12, m2, n / 2, q);
    Strassen(s1, c4, m3, n / 2, q);
    Strassen(s4, c5, m4, n / 2, q);
    Strassen(c6, s42, m5, n / 2, q);
    Strassen(c7, c8, m6, n / 2, q);
    Strassen(c9, c10, m7, n / 2, q); //递归过程
    add(m1, m4, s13, n / 2, q);
    sub(s13, m5, s13, n / 2, q);
    add(s13, m7, s13, n / 2, q);
    add(m3, m5, s23, n / 2, q);
    add(m2, m4, s33, n / 2, q);
    sub(m1, m2, s43, n / 2, q);
    add(s43, m3, s43, n / 2, q);
    add(s43, m6, s43, n / 2, q); //加减计算
    for (int i = 0; i < n / 2; i++) {
        for (int j = 0; j < n / 2; j++) {
            c[i * n + j] = s13[i * n / 2 + j];
            c[i * n + j + n / 2] = s23[i * n / 2 + j];
            c[(i + n / 2) * n + j] = s33[i * n / 2 + j];
            c[(i + n / 2) * n + j + n / 2] = s43[i * n / 2 + j];
        }
    } //矩阵合并，得到最终结果
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
    FILE* f;
    f = fopen("a.out","w");
    queue q(gpu_selector{});
    int* a = createMatrix(N);
    int* b = createMatrix(N);
    int* c = createMatrix(N);

    init(a, b, N);
    print(a, N,f);
    print(b, N,f);
    
    Strassen(a, b, c, N,q);
    print(c, N,f);
    fclose(f);
    delete[] a;
    delete[] b;
    delete[] c;
    
    return 0;
}