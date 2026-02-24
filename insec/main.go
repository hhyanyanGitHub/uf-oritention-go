package main

import (
	"fmt"
	"math"
)

// 已知相机参数 (Exterior Orientation)
type Camera struct {
	id                string
	omega, phi, kappa float64 // 姿态角 (弧度)
	xl, yl, zl        float64 // 相机绝对坐标 (比如国家大地坐标系)
	f                 float64 // 焦距 (毫米)
	m                 [3][3]float64
}

// 像点观测值 (在特定相机上的像素物理坐标 x, y)
type Observation struct {
	camIdx int     // 观测到该点的相机索引
	x, y   float64 // 照片上的坐标 (毫米)
}

func main() {
	fmt.Println("=== 摄影测量: 空间前方交会 (Space Intersection) ===")

	// 1. 设置已知的外方位元素 (我们之前的程序算出来的结果)
	cameras := []Camera{
		{
			id: "Cam_Left", f: 152.4,
			xl: 1000.0, yl: 1000.0, zl: 2000.0,
			omega: 0.015, phi: -0.012, kappa: 0.005, // 轻微倾斜
		},
		{
			id: "Cam_Right", f: 152.4,
			xl: 2000.0, yl: 1000.0, zl: 2000.0,
			omega: -0.010, phi: 0.020, kappa: -0.008,
		},
	}

	// 预先计算旋转矩阵
	for i := range cameras {
		calcRotationMatrix(&cameras[i])
	}

	// 2. 输入观测数据 (同一个地面点，在左右两张照片上的坐标)
	// (这些坐标是我用真实的数学反算出来的，所以理论上它们应该完美交会于 X=1500, Y=1200, Z=150)
	obs := []Observation{
		{camIdx: 0, x: 39.24150014636523, y: 13.925783508004365},  // 左相片上的坐标
		{camIdx: 1, x: -38.124045214887815, y: 17.62136773163286}, // 右相片上的坐标
	}

	// 3. 赋予地面点的初始近似值 (设在两台相机的正下方平面，Z=0)
	var X, Y, Z float64
	X = (cameras[0].xl + cameras[1].xl) / 2.0
	Y = (cameras[0].yl + cameras[1].yl) / 2.0
	Z = 0.0

	fmt.Printf("初始猜测坐标: X=%.2f, Y=%.2f, Z=%.2f\n\n", X, Y, Z)

	// 4. 最小二乘法非线性迭代
	var s0, s0old float64 = 1.0e30, 0
	iter := 0
	converged := false

	for !converged && iter < 10 {
		iter++
		s0old = s0

		// 共有 4 个观测方程式 (每个照片贡献 x 和 y)
		numEqs := len(obs) * 2
		J := make([][]float64, numEqs)
		for i := range J {
			J[i] = make([]float64, 3) // 未知数只有 3 个 (X, Y, Z)
		}
		E := make([]float64, numEqs) // 闭合差

		sumResSq := 0.0

		for i, ob := range obs {
			cam := cameras[ob.camIdx]

			// 计算从地面点到相机的向量差
			dx := X - cam.xl
			dy := Y - cam.yl
			dz := Z - cam.zl

			// 旋转到相机坐标系
			r := cam.m[0][0]*dx + cam.m[0][1]*dy + cam.m[0][2]*dz
			s := cam.m[1][0]*dx + cam.m[1][1]*dy + cam.m[1][2]*dz
			q := cam.m[2][0]*dx + cam.m[2][1]*dy + cam.m[2][2]*dz

			// 根据共线条件方程计算理论上的像点坐标
			calcX := -cam.f * r / q
			calcY := -cam.f * s / q

			// 计算闭合差 E = L_meas - L_calc
			epsX := ob.x - calcX
			epsY := ob.y - calcY

			rowX := 2 * i
			rowY := 2*i + 1
			E[rowX] = epsX
			E[rowY] = epsY
			sumResSq += epsX*epsX + epsY*epsY

			// 计算雅可比矩阵 J (偏导数矩阵)
			f_q2 := cam.f / (q * q)
			// 注意：这里是对地面点 X, Y, Z 求导，与之前对相机坐标 XL, YL, ZL 求导正好相差一个负号
			J[rowX][0] = f_q2 * (r*cam.m[2][0] - q*cam.m[0][0]) // d(x)/dX
			J[rowX][1] = f_q2 * (r*cam.m[2][1] - q*cam.m[0][1]) // d(x)/dY
			J[rowX][2] = f_q2 * (r*cam.m[2][2] - q*cam.m[0][2]) // d(x)/dZ

			J[rowY][0] = f_q2 * (s*cam.m[2][0] - q*cam.m[1][0]) // d(y)/dX
			J[rowY][1] = f_q2 * (s*cam.m[2][1] - q*cam.m[1][1]) // d(y)/dY
			J[rowY][2] = f_q2 * (s*cam.m[2][2] - q*cam.m[1][2]) // d(y)/dZ
		}

		// 计算单位权标准差 (中误差)
		s0 = math.Sqrt(sumResSq / float64(numEqs-3))
		fmt.Printf("迭代 %d: 单位权中误差 S0 = %.6f\n", iter, s0)

		if math.Abs(s0old-s0)/s0 < 1e-6 {
			converged = true
			break
		}

		// 求解法方程 N = J^T * J, U = J^T * E
		JT := transpose(J)
		N := multiplyMat(JT, J)
		U := multiplyMatVec(JT, E)
		delta := solveGaussian(N, U)

		// 更新地面点坐标
		X += delta[0]
		Y += delta[1]
		Z += delta[2]
	}

	// 5. 打印完美交会结果
	fmt.Println("\n=== 交会解算结果 (3D Ground Coordinates) ===")
	fmt.Printf("求得真实坐标:\n")
	fmt.Printf("  X = %10.4f (理论真值约为 1500.0000)\n", X)
	fmt.Printf("  Y = %10.4f (理论真值约为 1200.0000)\n", Y)
	fmt.Printf("  Z = %10.4f (理论真值约为  150.0000)\n", Z)

	fmt.Printf("\n完美收敛！这就是立体测图仪能提取高程 (DEM) 的核心数学原理。\n")
}

// ---------------------------------------------------------
// 矩阵及旋转运算辅助库
// ---------------------------------------------------------
func calcRotationMatrix(cam *Camera) {
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

func transpose(a [][]float64) [][]float64 {
	out := make([][]float64, len(a[0]))
	for i := range out {
		out[i] = make([]float64, len(a))
		for j := range out[i] {
			out[i][j] = a[j][i]
		}
	}
	return out
}

func multiplyMat(a, b [][]float64) [][]float64 {
	out := make([][]float64, len(a))
	for i := range out {
		out[i] = make([]float64, len(b[0]))
		for j := range out[i] {
			for k := range a[0] {
				out[i][j] += a[i][k] * b[k][j]
			}
		}
	}
	return out
}

func multiplyMatVec(a [][]float64, v []float64) []float64 {
	out := make([]float64, len(a))
	for i := range a {
		for k := range a[0] {
			out[i] += a[i][k] * v[k]
		}
	}
	return out
}

func solveGaussian(A [][]float64, B []float64) []float64 {
	n := len(B)
	mat := make([][]float64, n)
	for i := range mat {
		mat[i] = make([]float64, n+1)
		copy(mat[i][:n], A[i])
		mat[i][n] = B[i]
	}
	for i := 0; i < n; i++ {
		max := i
		for k := i + 1; k < n; k++ {
			if math.Abs(mat[k][i]) > math.Abs(mat[max][i]) {
				max = k
			}
		}
		mat[i], mat[max] = mat[max], mat[i]
		for j := i + 1; j < n; j++ {
			f := mat[j][i] / mat[i][i]
			for k := i; k <= n; k++ {
				mat[j][k] -= f * mat[i][k]
			}
		}
	}
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
