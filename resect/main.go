package main

import (
	"fmt"
	"math"
	"strings"
)

// 相机内外方位元素结构体
type CameraParam struct {
	omega, phi, kappa float64       // 旋转角 (弧度)
	xl, yl, zl        float64       // 相机中心坐标
	f                 float64       // 焦距
	m                 [3][3]float64 // 旋转矩阵
}

// 点坐标结构体
type Point struct {
	name       string
	x, y       float64 // 像点坐标 (毫米等)
	X, Y, Z    float64 // 物方地面坐标
	xres, yres float64 // 平差后的残差
}

func main() {
	// 1. 读取内置的测试数据 (原C代码注释中提供的样本)
	focalLength, points := loadTestData()
	numPts := len(points)
	fmt.Printf("载入数据完成: 焦距 %.3f, 控制点数量: %d\n", focalLength, numPts)

	// 2. 初始化相机参数
	cam := CameraParam{f: focalLength}

	// 3. 计算初始近似值 (极其重要，否则非线性迭代会发散)
	computeApproximations(&cam, points)
	fmt.Printf("初始近似值估算完成:\n  Omega=%.3f, Phi=%.3f, Kappa=%.3f\n  XL=%.3f, YL=%.3f, ZL=%.3f\n",
		cam.omega, cam.phi, cam.kappa, cam.xl, cam.yl, cam.zl)

	// 4. 开始最小二乘迭代
	var s0, s0old float64
	s0 = 1.0e30
	iter := 0
	converged := false

	fmt.Println("\n--- 开始空间后方交会迭代 ---")
	for !converged && iter < 15 {
		iter++
		s0old = s0

		// 计算当前的旋转矩阵
		calcRotationMatrix(&cam)

		// 构建雅可比矩阵 J 和 闭合差向量 E
		// 有 n 个点，每个点产生 x, y 两个观测方程，所以 J 是 (2n * 6) 的矩阵，E 是 (2n * 1) 的列向量
		J := make([][]float64, 2*numPts)
		for i := range J {
			J[i] = make([]float64, 6)
		}
		E := make([]float64, 2*numPts)

		sumResSq := 0.0 // 残差平方和

		// 填充 J 和 E
		for i, pt := range points {
			// 物方点到相机中心的坐标差
			dx := pt.X - cam.xl
			dy := pt.Y - cam.yl
			dz := pt.Z - cam.zl

			// 旋转到相机坐标系下的分量
			r := cam.m[0][0]*dx + cam.m[0][1]*dy + cam.m[0][2]*dz
			s := cam.m[1][0]*dx + cam.m[1][1]*dy + cam.m[1][2]*dz
			q := cam.m[2][0]*dx + cam.m[2][1]*dy + cam.m[2][2]*dz

			// 提取偏导数计算的公共因子
			f_q2 := cam.f / (q * q)
			m33dy_m32dz := -cam.m[2][2]*dy + cam.m[2][1]*dz
			m13dy_m12dz := -cam.m[0][2]*dy + cam.m[0][1]*dz
			m23dy_m22dz := -cam.m[1][2]*dy + cam.m[1][1]*dz

			// 提前计算角度的三角函数 (用作导数)
			so, co := math.Sin(cam.omega), math.Cos(cam.omega)
			sp, cp := math.Sin(cam.phi), math.Cos(cam.phi)
			sk, ck := math.Sin(cam.kappa), math.Cos(cam.kappa)

			// 填写第 i 个点的 x 观测方程 (对应矩阵的 2i 行)
			rowX := 2 * i
			J[rowX][0] = f_q2 * (r*(m33dy_m32dz) - q*(m13dy_m12dz))
			J[rowX][1] = f_q2 * (r*(cp*dx+so*sp*dy-co*sp*dz) - q*(-sp*ck*dx+so*cp*ck*dy-co*cp*ck*dz))
			J[rowX][2] = -cam.f * s / q
			J[rowX][3] = -cam.f * (r*cam.m[2][0] - q*cam.m[0][0]) / (q * q)
			J[rowX][4] = -cam.f * (r*cam.m[2][1] - q*cam.m[0][1]) / (q * q)
			J[rowX][5] = -cam.f * (r*cam.m[2][2] - q*cam.m[0][2]) / (q * q)

			// 填写第 i 个点的 y 观测方程 (对应矩阵的 2i+1 行)
			rowY := 2*i + 1
			J[rowY][0] = f_q2 * (s*(m33dy_m32dz) - q*(m23dy_m22dz))
			J[rowY][1] = f_q2 * (s*(cp*dx+so*sp*dy-co*sp*dz) - q*(sp*sk*dx-so*cp*sk*dy+co*cp*sk*dz))
			J[rowY][2] = cam.f * r / q
			J[rowY][3] = -cam.f * (s*cam.m[2][0] - q*cam.m[1][0]) / (q * q)
			J[rowY][4] = -cam.f * (s*cam.m[2][1] - q*cam.m[1][1]) / (q * q)
			J[rowY][5] = -cam.f * (s*cam.m[2][2] - q*cam.m[1][2]) / (q * q)

			// 计算理论值与实际测量值的闭合差 epsilon
			epsX := pt.x + cam.f*r/q
			epsY := pt.y + cam.f*s/q
			E[rowX] = epsX
			E[rowY] = epsY

			sumResSq += epsX*epsX + epsY*epsY

			// 保存残差供输出
			points[i].xres = -epsX
			points[i].yres = -epsY
		}

		// 计算单位权标准差
		s0 = math.Sqrt(sumResSq / float64(2*numPts-6))
		fmt.Printf("  迭代 %d, S0 (单位权中误差) = %.6f\n", iter, s0)

		// 检查收敛
		if math.Abs(s0old-s0)/s0 < 0.0001 {
			converged = true
			break
		}

		// 核心平差计算：法方程 N = J^T * J,  U = J^T * E
		JT := transpose(J)
		N := multiplyMat(JT, J)
		U := multiplyMatVec(JT, E)

		// 求解法方程：N * Delta = U  => Delta = N^{-1} * U
		delta := solveGaussian(N, U)

		// 将改正数应用到参数上
		cam.omega += delta[0]
		cam.phi += delta[1]
		cam.kappa += delta[2]
		cam.xl += delta[3]
		cam.yl += delta[4]
		cam.zl += delta[5]
	}

	// 5. 输出最终结果
	fmt.Println("\n--- 最终计算结果 ---")
	fmt.Printf("Omega = %10.4f 度\n", cam.omega*180/math.Pi)
	fmt.Printf("Phi   = %10.4f 度\n", cam.phi*180/math.Pi)
	fmt.Printf("Kappa = %10.4f 度\n", cam.kappa*180/math.Pi)
	fmt.Printf("XL    = %10.4f\n", cam.xl)
	fmt.Printf("YL    = %10.4f\n", cam.yl)
	fmt.Printf("ZL    = %10.4f\n", cam.zl)

	fmt.Println("\n像点残差 (理论与实际像点位置的微小差异):")
	for _, p := range points {
		fmt.Printf("  点 %-6s : dx = %7.4f, dy = %7.4f\n", p.name, p.xres, p.yres)
	}
}

// ----------------------------------------------------
// 以下为核心数学函数与辅助函数，还原了C源码中优美的逻辑
// ----------------------------------------------------

func calcRotationMatrix(cam *CameraParam) {
	so, co := math.Sin(cam.omega), math.Cos(cam.omega)
	sp, cp := math.Sin(cam.phi), math.Cos(cam.phi)
	sk, ck := math.Sin(cam.kappa), math.Cos(cam.kappa)

	cam.m[0][0] = cp * ck
	cam.m[0][1] = so*sp*ck + co*sk
	cam.m[0][2] = -co*sp*ck + so*sk
	cam.m[1][0] = -cp * sk
	cam.m[1][1] = -so*sp*sk + co*ck
	cam.m[1][2] = co*sp*sk + so*ck
	cam.m[2][0] = sp
	cam.m[2][1] = -so * cp
	cam.m[2][2] = co * cp
}

// 估算初始参数 (高度估算 & 2D仿射变换)
func computeApproximations(cam *CameraParam, points []Point) {
	n := len(points)
	sumH := 0.0
	count := 0

	// 1. 估算航高 (ZL)
	for i := 0; i < n-1; i++ {
		for j := i + 1; j < n; j++ {
			dx := points[j].x - points[i].x
			dy := points[j].y - points[i].y
			cx := points[i].x*points[i].Z - points[j].x*points[j].Z
			cy := points[i].y*points[i].Z - points[j].y*points[j].Z
			dij2 := (points[j].X-points[i].X)*(points[j].X-points[i].X) + (points[j].Y-points[i].Y)*(points[j].Y-points[i].Y)

			a := dx*dx + dy*dy
			b := 2.0 * (dx*cx + dy*cy)
			c := cx*cx + cy*cy - cam.f*cam.f*dij2

			sqrtterm := b*b - 4*a*c
			if sqrtterm >= 0 {
				H := (-b + math.Sqrt(sqrtterm)) / (2 * a)
				sumH += H
				count++
			}
		}
	}
	cam.zl = sumH / float64(count)
	cam.omega, cam.phi = 0, 0

	// 2. 利用 2D 共形变换估算 XL, YL 和 Kappa
	A := make([][]float64, 4)
	for i := range A {
		A[i] = make([]float64, 4)
	}
	L := make([]float64, 4)

	for _, pt := range points {
		xvert := pt.x * (cam.zl - pt.Z) / cam.f
		yvert := pt.y * (cam.zl - pt.Z) / cam.f

		A[0][0] += xvert*xvert + yvert*yvert
		A[0][2] += xvert
		A[0][3] += yvert
		L[0] += xvert*pt.X + yvert*pt.Y
		L[1] += xvert*pt.Y - yvert*pt.X
		L[2] += pt.X
		L[3] += pt.Y
	}
	A[1][1] = A[0][0]
	A[1][2] = -A[0][3]
	A[1][3] = A[0][2]
	A[2][2] = float64(n)
	A[3][3] = float64(n)

	// 对称补充
	A[2][0] = A[0][2]
	A[3][0] = A[0][3]
	A[2][1] = A[1][2]
	A[3][1] = A[1][3]

	sol := solveGaussian(A, L)
	cam.kappa = math.Atan2(sol[1], sol[0])
	cam.xl = sol[2]
	cam.yl = sol[3]
}

// 简单的矩阵转置
func transpose(a [][]float64) [][]float64 {
	rows, cols := len(a), len(a[0])
	out := make([][]float64, cols)
	for i := range out {
		out[i] = make([]float64, rows)
		for j := range out[i] {
			out[i][j] = a[j][i]
		}
	}
	return out
}

// 矩阵乘法 C = A * B
func multiplyMat(a, b [][]float64) [][]float64 {
	rowsA, colsA := len(a), len(a[0])
	colsB := len(b[0])
	out := make([][]float64, rowsA)
	for i := 0; i < rowsA; i++ {
		out[i] = make([]float64, colsB)
		for j := 0; j < colsB; j++ {
			sum := 0.0
			for k := 0; k < colsA; k++ {
				sum += a[i][k] * b[k][j]
			}
			out[i][j] = sum
		}
	}
	return out
}

// 矩阵乘向量 C = A * V
func multiplyMatVec(a [][]float64, v []float64) []float64 {
	rowsA, colsA := len(a), len(a[0])
	out := make([]float64, rowsA)
	for i := 0; i < rowsA; i++ {
		sum := 0.0
		for k := 0; k < colsA; k++ {
			sum += a[i][k] * v[k]
		}
		out[i] = sum
	}
	return out
}

// 高斯消元法解线性方程组 A*x = B
func solveGaussian(A [][]float64, B []float64) []float64 {
	n := len(B)
	// 拷贝矩阵防止原地修改
	mat := make([][]float64, n)
	for i := range mat {
		mat[i] = make([]float64, n+1)
		copy(mat[i][:n], A[i])
		mat[i][n] = B[i]
	}

	for i := 0; i < n; i++ {
		// 向前消元
		for j := i + 1; j < n; j++ {
			factor := mat[j][i] / mat[i][i]
			for k := i; k <= n; k++ {
				mat[j][k] -= factor * mat[i][k]
			}
		}
	}

	// 回代求解
	x := make([]float64, n)
	for i := n - 1; i >= 0; i-- {
		sum := mat[i][n]
		for j := i + 1; j < n; j++ {
			sum -= mat[i][j] * x[j]
		}
		x[i] = sum / mat[i][i]
	}
	return x
}

// 载入测试数据 (原C代码中的样本数据)
func loadTestData() (float64, []Point) {
	data := `
152.916
tn08   86.421  -83.977  1268.1022  1455.0274  -14.3939
ts08 -100.916   92.582   732.1811   545.3437  -14.7009
re08  -98.322  -89.161  1454.5532   731.6659  -14.3509
rw08   78.812   98.123   545.2449  1268.2324  -14.6639
0000   -8.641    5.630  1000.0000  1000.0000  -14.4540
`
	lines := strings.Split(strings.TrimSpace(data), "\n")
	var f float64
	fmt.Sscanf(lines[0], "%f", &f)

	var pts []Point
	for i := 1; i < len(lines); i++ {
		var p Point
		fmt.Sscanf(lines[i], "%s %f %f %f %f %f", &p.name, &p.x, &p.y, &p.X, &p.Y, &p.Z)
		pts = append(pts, p)
	}
	return f, pts
}
