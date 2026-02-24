package main

import (
	"fmt"
	"math"
	"strings"
)

// 摄影机参数
type CameraParam struct {
	omega, phi, kappa float64 // 旋转角
	xl, yl, zl        float64 // 空间坐标
	f                 float64 // 焦距
	m                 [3][3]float64
}

// 立体像对中的同名点
type Point struct {
	name                           string
	xleft, yleft                   float64 // 左片像素坐标
	xright, yright                 float64 // 右片像素坐标
	X, Y, Z                        float64 // 迭代求解的 3D 模型坐标
	xl_res, yl_res, xr_res, yr_res float64 // 最终残差
}

func main() {
	// 1. 载入内置的相对定向测试数据
	focalLength, points := loadRelOrData()
	numPts := len(points)
	fmt.Printf("载入立体数据完成: 焦距 %.3f, 同名点数量: %d\n", focalLength, numPts)

	// 总未知数数量 = 5(右相机的 omega, phi, kappa, YL, ZL) + 3*n (每个点的 X,Y,Z)
	numUnknowns := 5 + 3*numPts

	// 2. 初始化基准 (极其关键)
	leftCam := CameraParam{f: focalLength, zl: focalLength}
	rightCam := CameraParam{f: focalLength, zl: focalLength}

	// 估算摄影基线 b (左右相机在 X 轴的距离)，用视差平均值计算
	parallaxSum := 0.0
	for i := range points {
		parallaxSum += points[i].xleft - points[i].xright
		// 初始 3D 模型坐标：X, Y 借用左片坐标，Z 初始为 0
		points[i].X = points[i].xleft
		points[i].Y = points[i].yleft
		points[i].Z = 0.0
	}
	rightCam.xl = parallaxSum / float64(numPts) // 锁定不参与迭代

	// 左相机始终固定，旋转矩阵永远是单位阵
	calcRotationMatrix(&leftCam)

	// 3. 开始光束法平差迭代
	var s0, s0old float64 = 1.0e35, 0
	iter := 0
	converged := false

	fmt.Printf("初始右相机位置: XL=%.3f (固定), YL=%.3f, ZL=%.3f\n", rightCam.xl, rightCam.yl, rightCam.zl)
	fmt.Println("\n--- 开始相对定向(微型光束法)迭代 ---")

	for !converged && iter < 15 {
		iter++
		s0old = s0
		calcRotationMatrix(&rightCam)

		// 雅可比矩阵 J: 行数 = 4 * numPts (每个点左2个方程，右2个方程)
		// 列数 = 5(右相机) + 3*numPts(3D点)
		numEqs := 4 * numPts
		J := make([][]float64, numEqs)
		for i := range J {
			J[i] = make([]float64, numUnknowns)
		}
		E := make([]float64, numEqs) // 闭合差向量
		sumResSq := 0.0

		// 逐点建立观测方程
		for i, pt := range points {
			// ------------ 左相机的方程 (只对点的 X,Y,Z 有偏导数) ------------
			r, s, q, lpartials := calcDerivatives(leftCam, pt.X, pt.Y, pt.Z)

			rowLx := 4 * i
			rowLy := 4*i + 1
			// 对物体坐标 X,Y,Z 的偏导数 = 负的对相机 XL,YL,ZL 的偏导数
			// 填入雅可比矩阵对应的位置 (列索引从 5 + 3*i 开始)
			colPt := 5 + 3*i
			J[rowLx][colPt+0] = -lpartials[0][3] // d(xl)/dX
			J[rowLx][colPt+1] = -lpartials[0][4] // d(xl)/dY
			J[rowLx][colPt+2] = -lpartials[0][5] // d(xl)/dZ

			J[rowLy][colPt+0] = -lpartials[1][3] // d(yl)/dX
			J[rowLy][colPt+1] = -lpartials[1][4] // d(yl)/dY
			J[rowLy][colPt+2] = -lpartials[1][5] // d(yl)/dZ

			epsLx := pt.xleft + leftCam.f*r/q
			epsLy := pt.yleft + leftCam.f*s/q
			E[rowLx] = epsLx
			E[rowLy] = epsLy
			sumResSq += epsLx*epsLx + epsLy*epsLy
			points[i].xl_res, points[i].yl_res = -epsLx, -epsLy

			// ------------ 右相机的方程 (对相机姿态和点 XYZ 都有偏导数) ------------
			r, s, q, rpartials := calcDerivatives(rightCam, pt.X, pt.Y, pt.Z)

			rowRx := 4*i + 2
			rowRy := 4*i + 3

			// 1. 对右相机的 5 个未知数 (omega, phi, kappa, YL, ZL) 求偏导
			// 列索引 0, 1, 2, 3, 4
			J[rowRx][0] = rpartials[0][0] // omega
			J[rowRx][1] = rpartials[0][1] // phi
			J[rowRx][2] = rpartials[0][2] // kappa
			J[rowRx][3] = rpartials[0][4] // YL (跳过 XL)
			J[rowRx][4] = rpartials[0][5] // ZL

			J[rowRy][0] = rpartials[1][0]
			J[rowRy][1] = rpartials[1][1]
			J[rowRy][2] = rpartials[1][2]
			J[rowRy][3] = rpartials[1][4]
			J[rowRy][4] = rpartials[1][5]

			// 2. 对 3D 物体坐标 X, Y, Z 求偏导
			J[rowRx][colPt+0] = -rpartials[0][3]
			J[rowRx][colPt+1] = -rpartials[0][4]
			J[rowRx][colPt+2] = -rpartials[0][5]

			J[rowRy][colPt+0] = -rpartials[1][3]
			J[rowRy][colPt+1] = -rpartials[1][4]
			J[rowRy][colPt+2] = -rpartials[1][5]

			epsRx := pt.xright + rightCam.f*r/q
			epsRy := pt.yright + rightCam.f*s/q
			E[rowRx] = epsRx
			E[rowRy] = epsRy
			sumResSq += epsRx*epsRx + epsRy*epsRy
			points[i].xr_res, points[i].yr_res = -epsRx, -epsRy
		}

		// 计算单位权标准差 (自由度 = 观测值总数 - 未知数总数)
		dof := float64(numEqs - numUnknowns)
		s0 = math.Sqrt(sumResSq / dof)
		fmt.Printf("  迭代 %d, S0 = %.6f\n", iter, s0)

		if math.Abs(s0old-s0)/s0 < 0.0001 {
			converged = true
			break
		}

		// 组建并求解巨型法方程：N = J^T * J, U = J^T * E
		JT := transpose(J)
		N := multiplyMat(JT, J)
		U := multiplyMatVec(JT, E)
		delta := solveGaussian(N, U)

		// 将改正数应用到右相机
		rightCam.omega += delta[0]
		rightCam.phi += delta[1]
		rightCam.kappa += delta[2]
		rightCam.yl += delta[3]
		rightCam.zl += delta[4]

		// 将改正数应用到所有 3D 点
		for i := range points {
			points[i].X += delta[5+3*i]
			points[i].Y += delta[5+3*i+1]
			points[i].Z += delta[5+3*i+2]
		}
	}

	// 4. 输出完美的重建成品
	fmt.Println("\n--- 相对定向(模型重建) 最终结果 ---")
	fmt.Printf("右相机姿态:\n  Omega = %8.4f 度\n  Phi   = %8.4f 度\n  Kappa = %8.4f 度\n",
		rightCam.omega*180/math.Pi, rightCam.phi*180/math.Pi, rightCam.kappa*180/math.Pi)
	fmt.Printf("右相机位置:\n  XL    = %8.4f (摄影基线, 锁定)\n  YL    = %8.4f\n  ZL    = %8.4f\n",
		rightCam.xl, rightCam.yl, rightCam.zl)

	fmt.Println("\n建立的空间三维模型坐标 (相对坐标):")
	fmt.Printf("%8s %9s %9s %9s\n", "点号", "X", "Y", "Z")
	for _, p := range points {
		fmt.Printf("%8s %9.4f %9.4f %9.4f\n", p.name, p.X, p.Y, p.Z)
	}
}

// ----------------------------------------------------
// 底层数学运算库 (与后方交会类似)
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

// 计算 r, s, q 以及 6个外方位元素的偏导数矩阵 [2][6]
func calcDerivatives(cam CameraParam, X, Y, Z float64) (float64, float64, float64, [2][6]float64) {
	dx := X - cam.xl
	dy := Y - cam.yl
	dz := Z - cam.zl

	r := cam.m[0][0]*dx + cam.m[0][1]*dy + cam.m[0][2]*dz
	s := cam.m[1][0]*dx + cam.m[1][1]*dy + cam.m[1][2]*dz
	q := cam.m[2][0]*dx + cam.m[2][1]*dy + cam.m[2][2]*dz

	f_q2 := cam.f / (q * q)
	so, co := math.Sin(cam.omega), math.Cos(cam.omega)
	sp, cp := math.Sin(cam.phi), math.Cos(cam.phi)
	sk, ck := math.Sin(cam.kappa), math.Cos(cam.kappa)

	var p [2][6]float64
	// dx方程的导数
	p[0][0] = f_q2 * (r*(-cam.m[2][2]*dy+cam.m[2][1]*dz) - q*(-cam.m[0][2]*dy+cam.m[0][1]*dz))
	p[0][1] = f_q2 * (r*(cp*dx+so*sp*dy-co*sp*dz) - q*(-sp*ck*dx+so*cp*ck*dy-co*cp*ck*dz))
	p[0][2] = -cam.f * s / q
	p[0][3] = -cam.f * (r*cam.m[2][0] - q*cam.m[0][0]) / (q * q) // XL
	p[0][4] = -cam.f * (r*cam.m[2][1] - q*cam.m[0][1]) / (q * q) // YL
	p[0][5] = -cam.f * (r*cam.m[2][2] - q*cam.m[0][2]) / (q * q) // ZL

	// dy方程的导数
	p[1][0] = f_q2 * (s*(-cam.m[2][2]*dy+cam.m[2][1]*dz) - q*(-cam.m[1][2]*dy+cam.m[1][1]*dz))
	p[1][1] = f_q2 * (s*(cp*dx+so*sp*dy-co*sp*dz) - q*(sp*sk*dx-so*cp*sk*dy+co*cp*sk*dz))
	p[1][2] = cam.f * r / q
	p[1][3] = -cam.f * (s*cam.m[2][0] - q*cam.m[1][0]) / (q * q) // XL
	p[1][4] = -cam.f * (s*cam.m[2][1] - q*cam.m[1][1]) / (q * q) // YL
	p[1][5] = -cam.f * (s*cam.m[2][2] - q*cam.m[1][2]) / (q * q) // ZL

	return r, s, q, p
}

// 矩阵辅助函数
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

func multiplyMat(a, b [][]float64) [][]float64 {
	rowsA, colsA, colsB := len(a), len(a[0]), len(b[0])
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

// 求解线性方程组 (带简单的主元选取以保证数值稳定)
func solveGaussian(A [][]float64, B []float64) []float64 {
	n := len(B)
	mat := make([][]float64, n)
	for i := range mat {
		mat[i] = make([]float64, n+1)
		copy(mat[i][:n], A[i])
		mat[i][n] = B[i]
	}

	for i := 0; i < n; i++ {
		// 主元选取
		maxRow := i
		for k := i + 1; k < n; k++ {
			if math.Abs(mat[k][i]) > math.Abs(mat[maxRow][i]) {
				maxRow = k
			}
		}
		mat[i], mat[maxRow] = mat[maxRow], mat[i]

		for j := i + 1; j < n; j++ {
			factor := mat[j][i] / mat[i][i]
			for k := i; k <= n; k++ {
				mat[j][k] -= factor * mat[i][k]
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

// 载入 C 源码中的原版样本数据
func loadRelOrData() (float64, []Point) {
	data := `
152.113
a -4.878   1.974  -97.920  -2.923
b 89.307   2.709   -1.507  -1.856
c  0.261  84.144  -90.917  78.970
d 90.334  83.843   -1.571  79.470
e -4.668 -86.821 -100.060 -95.748
f 88.599 -85.274   -0.965 -94.319
`
	lines := strings.Split(strings.TrimSpace(data), "\n")
	var f float64
	fmt.Sscanf(lines[0], "%f", &f)

	var pts []Point
	for i := 1; i < len(lines); i++ {
		var p Point
		fmt.Sscanf(lines[i], "%s %f %f %f %f", &p.name, &p.xleft, &p.yleft, &p.xright, &p.yright)
		pts = append(pts, p)
	}
	return f, pts
}
