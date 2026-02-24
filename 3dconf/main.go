// 3D 相似变换 (绝对定向) 的最小二乘法平差实现
// 适用于 GIS 坐标系转换、地理配准等场景
// 通过七参数模型 (比例尺、旋转角、平移量) 将相对模型坐标转换到真实世界坐标系
// 主要步骤包括：
// 1. 载入公共控制点和未知点数据
// 2. 使用 Dewitt (1996) 的几何最强三角形法估算初始变换参数
// 3. 迭代进行最小二乘法平差，更新变换参数直到收敛
// 4. 输出最终变换参数及其精度评定
// 5. 应用最终变换参数将未知点转换到真实世界坐标系，并计算其精度
// 代码实现中包含了核心的数学库函数，如雅可比矩阵构建、矩阵运算、Gaussian 消元法求解线性方程组等

// abs_ori.go

package main

import (
	"fmt"
	"math"
	"strings"
)

// 公共点 (已知在相对模型和真实世界中的双重坐标)
type CommonPoint struct {
	name             string
	xarb, yarb, zarb float64 // 任意坐标 (Arbitrary) - 即上一步的相对定向输出
	xcon, ycon, zcon float64 // 控制坐标 (Control) - 即真实的 GIS 坐标
	xres, yres, zres float64 // 平差残差
}

// 未知点 (仅在相对模型中已知，需要被转换到真实世界)
type UnknownPoint struct {
	name    string
	x, y, z float64
}

// 七参数转换模型
type TransformParams struct {
	scale             float64
	omega, phi, kappa float64
	tx, ty, tz        float64
	m                 [3][3]float64 // 旋转矩阵缓存
}

func main() {
	// 1. 载入内置测试数据
	comPts, unkPts := loadAbsOrData()
	fmt.Printf("载入数据完成: 公共控制点 %d 个, 待转换未知点 %d 个\n", len(comPts), len(unkPts))

	var params TransformParams

	// 2. 使用 Dewitt (1996) 的几何最强三角形法估算初始值
	initApproximations(&params, comPts)

	// 3. 最小二乘法平差迭代
	var s0, s0old float64 = 1.0e30, 0
	iter := 0
	converged := false
	var N_inv [][]float64 // 用于保存最后一次收敛的法方程逆矩阵 (协方差矩阵)

	fmt.Println("\n--- 开始 3D 相似变换 (绝对定向) 迭代 ---")
	for !converged && iter < 15 {
		iter++
		s0old = s0
		calcRotationMatrixOPK(&params)

		// 构建雅可比矩阵 J 和闭合差 E
		// 有 n 个公共点，每个点产生 3 个坐标方程 (X, Y, Z)，共 3n 行，未知数 7 列
		numEqs := 3 * len(comPts)
		J := make([][]float64, numEqs)
		for i := range J {
			J[i] = make([]float64, 7)
		}
		E := make([]float64, numEqs)
		sumResSq := 0.0

		for i, pt := range comPts {
			// 计算雅可比矩阵中的一行 (3x7 偏导数)
			AI := formJacobianAI(params, pt.xarb, pt.yarb, pt.zarb)

			// 计算理论值与实际控制坐标的差值
			// X_calc = S * (m11*x + m21*y + m31*z) + Tx
			// 这里利用了偏导数矩阵的第一列其实刚好等于 R * X_arb
			calcX := params.scale*AI[0][0] + params.tx
			calcY := params.scale*AI[1][0] + params.ty
			calcZ := params.scale*AI[2][0] + params.tz

			epsX := pt.xcon - calcX
			epsY := pt.ycon - calcY
			epsZ := pt.zcon - calcZ

			row := 3 * i
			for k := 0; k < 7; k++ {
				J[row+0][k] = AI[0][k]
				J[row+1][k] = AI[1][k]
				J[row+2][k] = AI[2][k]
			}

			E[row+0] = epsX
			E[row+1] = epsY
			E[row+2] = epsZ

			sumResSq += epsX*epsX + epsY*epsY + epsZ*epsZ

			// 记录残差 (真实值 - 计算值)
			comPts[i].xres = epsX
			comPts[i].yres = epsY
			comPts[i].zres = epsZ
		}

		// 计算单位权标准差
		s0 = math.Sqrt(sumResSq / float64(numEqs-7))
		fmt.Printf("  迭代 %d, S0 = %.6f\n", iter, s0)

		if math.Abs(s0old-s0)/s0 < 0.0001 {
			converged = true
			// 收敛时，求法方程的逆矩阵以用于精度评定
			JT := transpose(J)
			N := multiplyMat(JT, J)
			N_inv = inverseMat(N)
			break
		}

		// 法方程求解: N * Delta = U
		JT := transpose(J)
		N := multiplyMat(JT, J)
		U := multiplyMatVec(JT, E)
		delta := solveGaussian(N, U)

		// 更新参数
		params.scale += delta[0]
		params.omega += delta[1]
		params.phi += delta[2]
		params.kappa += delta[3]
		params.tx += delta[4]
		params.ty += delta[5]
		params.tz += delta[6]
	}

	// 4. 打印最终变换参数及标准差
	fmt.Println("\n--- 绝对定向(七参数) 最终平差结果 ---")
	fmt.Printf("Scale = %10.5f  ± %6.5f\n", params.scale, s0*math.Sqrt(N_inv[0][0]))
	fmt.Printf("Omega = %10.4f° ± %6.4f°\n", params.omega*180/math.Pi, s0*math.Sqrt(N_inv[1][1])*180/math.Pi)
	fmt.Printf("Phi   = %10.4f° ± %6.4f°\n", params.phi*180/math.Pi, s0*math.Sqrt(N_inv[2][2])*180/math.Pi)
	fmt.Printf("Kappa = %10.4f° ± %6.4f°\n", params.kappa*180/math.Pi, s0*math.Sqrt(N_inv[3][3])*180/math.Pi)
	fmt.Printf("Tx    = %10.3f  ± %6.3f\n", params.tx, s0*math.Sqrt(N_inv[4][4]))
	fmt.Printf("Ty    = %10.3f  ± %6.3f\n", params.ty, s0*math.Sqrt(N_inv[5][5]))
	fmt.Printf("Tz    = %10.3f  ± %6.3f\n", params.tz, s0*math.Sqrt(N_inv[6][6]))

	// 5. 应用七参数：将未知点转换到真实的地球坐标系！(带严格误差传播)
	fmt.Println("\n\n--- GIS 坐标系转换结果 (Transformed Unknown Points) ---")
	fmt.Printf("%8s %9s %9s %9s %7s %7s %7s\n", "点号", "真实 X", "真实 Y", "真实 Z", "SDev.X", "SDev.Y", "SDev.Z")

	for _, pt := range unkPts {
		// 获取这个点在当前参数下的雅可比偏导矩阵 AI (3x7)
		AI := formJacobianAI(params, pt.x, pt.y, pt.z)

		// 坐标转换
		realX := params.scale*AI[0][0] + params.tx
		realY := params.scale*AI[1][0] + params.ty
		realZ := params.scale*AI[2][0] + params.tz

		// 极其硬核的误差传播定律：Q_xyz = AI * N_inv * AI^T
		// 用于计算这个新生成的点在 X,Y,Z 方向上的中误差 (点位精度)
		AIQ := multiplyMat(AI, N_inv)
		AIT := transpose(AI)
		Q_xyz := multiplyMat(AIQ, AIT)

		sdX := s0 * math.Sqrt(Q_xyz[0][0])
		sdY := s0 * math.Sqrt(Q_xyz[1][1])
		sdZ := s0 * math.Sqrt(Q_xyz[2][2])

		fmt.Printf("%8s %9.3f %9.3f %9.3f %7.3f %7.3f %7.3f\n", pt.name, realX, realY, realZ, sdX, sdY, sdZ)
	}
}

// -------------------------------------------------------------------------
// 核心数学库：初值估算、偏导数求取与矩阵运算
// -------------------------------------------------------------------------

// Dewitt (1996) 初值估算法
func initApproximations(p *TransformParams, comPts []CommonPoint) {
	n := len(comPts)

	// 1. 估算比例尺 (两套坐标系中各点连线距离比值的平均)
	sumScale := 0.0
	numScale := 0
	for i := 0; i < n-1; i++ {
		for j := i + 1; j < n; j++ {
			dc := math.Sqrt(sq(comPts[i].xcon-comPts[j].xcon) + sq(comPts[i].ycon-comPts[j].ycon) + sq(comPts[i].zcon-comPts[j].zcon))
			da := math.Sqrt(sq(comPts[i].xarb-comPts[j].xarb) + sq(comPts[i].yarb-comPts[j].yarb) + sq(comPts[i].zarb-comPts[j].zarb))
			sumScale += dc / da
			numScale++
		}
	}
	p.scale = sumScale / float64(numScale)

	// 2. 寻找几何最强三角形 (高最大)
	maxAltSq := 0.0
	var pt1, pt2, pt3 int
	for i := 0; i < n-2; i++ {
		for j := i + 1; j < n-1; j++ {
			d12Sq := sq(comPts[i].xcon-comPts[j].xcon) + sq(comPts[i].ycon-comPts[j].ycon) + sq(comPts[i].zcon-comPts[j].zcon)
			for k := j + 1; k < n; k++ {
				d13Sq := sq(comPts[i].xcon-comPts[k].xcon) + sq(comPts[i].ycon-comPts[k].ycon) + sq(comPts[i].zcon-comPts[k].zcon)
				d23Sq := sq(comPts[j].xcon-comPts[k].xcon) + sq(comPts[j].ycon-comPts[k].ycon) + sq(comPts[j].zcon-comPts[k].zcon)

				var a2, b2, c2 float64
				if d12Sq >= d13Sq && d12Sq >= d23Sq {
					c2, a2, b2 = d12Sq, d13Sq, d23Sq
				} else if d13Sq >= d12Sq && d13Sq >= d23Sq {
					c2, a2, b2 = d13Sq, d12Sq, d23Sq
				} else {
					c2, a2, b2 = d23Sq, d12Sq, d13Sq
				}

				// 海伦公式推导的高的平方
				h2 := (2*(c2*(a2+b2)+a2*b2) - a2*a2 - b2*b2 - c2*c2) / (4 * c2)
				if h2 > maxAltSq {
					maxAltSq = h2
					pt1, pt2, pt3 = i, j, k
				}
			}
		}
	}
	fmt.Printf("初值估算: 找到最强三角形，由控制点 %s, %s, %s 构成\n", comPts[pt1].name, comPts[pt2].name, comPts[pt3].name)

	// 利用最强三角形求平面的倾角(Tilt)、方位角(Azimuth)和旋角(Swing)，最终求得总体旋转矩阵
	// 详见原著 3D相似变换章节...
	A_arb, B_arb, C_arb := calcPlaneCoeffs(comPts[pt1].xarb, comPts[pt1].yarb, comPts[pt1].zarb,
		comPts[pt2].xarb, comPts[pt2].yarb, comPts[pt2].zarb, comPts[pt3].xarb, comPts[pt3].yarb, comPts[pt3].zarb)

	A_con, B_con, C_con := calcPlaneCoeffs(comPts[pt1].xcon, comPts[pt1].ycon, comPts[pt1].zcon,
		comPts[pt2].xcon, comPts[pt2].ycon, comPts[pt2].zcon, comPts[pt3].xcon, comPts[pt3].ycon, comPts[pt3].zcon)

	arbTilt := math.Atan2(C_arb, math.Hypot(A_arb, B_arb)) + math.Pi/2
	arbAz := math.Atan2(A_arb, B_arb)
	conTilt := math.Atan2(C_con, math.Hypot(A_con, B_con)) + math.Pi/2
	conAz := math.Atan2(A_con, B_con)

	arbRotBase := getRotMatTSA(arbTilt, 0, arbAz)
	conRotMat := getRotMatTSA(conTilt, 0, conAz)

	x_arb1, y_arb1 := applyRot2D(arbRotBase, comPts[pt1].xarb, comPts[pt1].yarb, comPts[pt1].zarb)
	x_arb2, y_arb2 := applyRot2D(arbRotBase, comPts[pt2].xarb, comPts[pt2].yarb, comPts[pt2].zarb)
	x_con1, y_con1 := applyRot2D(conRotMat, comPts[pt1].xcon, comPts[pt1].ycon, comPts[pt1].zcon)
	x_con2, y_con2 := applyRot2D(conRotMat, comPts[pt2].xcon, comPts[pt2].ycon, comPts[pt2].zcon)

	azCon := math.Atan2(x_con2-x_con1, y_con2-y_con1)
	azArb := math.Atan2(x_arb2-x_arb1, y_arb2-y_arb1)
	swing := azCon - azArb

	arbRotMat := getRotMatTSA(arbTilt, swing, arbAz)
	// Full Rot = ConRot^T * ArbRot
	conRotT := transpose(conRotMat)
	fullRotMat := multiplyMat(conRotT, arbRotMat)

	// 从总旋转矩阵中提取 omega, phi, kappa
	p.phi = math.Asin(fullRotMat[2][0])
	p.omega = math.Atan2(-fullRotMat[2][1], fullRotMat[2][2])
	p.kappa = math.Atan2(-fullRotMat[1][0], fullRotMat[0][0])
	p.m = [3][3]float64{
		{fullRotMat[0][0], fullRotMat[0][1], fullRotMat[0][2]},
		{fullRotMat[1][0], fullRotMat[1][1], fullRotMat[1][2]},
		{fullRotMat[2][0], fullRotMat[2][1], fullRotMat[2][2]},
	}

	// 3. 估算平移量 (质心对齐)
	var txSum, tySum, tzSum float64
	for _, pt := range comPts {
		rx := p.m[0][0]*pt.xarb + p.m[1][0]*pt.yarb + p.m[2][0]*pt.zarb
		ry := p.m[0][1]*pt.xarb + p.m[1][1]*pt.yarb + p.m[2][1]*pt.zarb
		rz := p.m[0][2]*pt.xarb + p.m[1][2]*pt.yarb + p.m[2][2]*pt.zarb
		txSum += pt.xcon - p.scale*rx
		tySum += pt.ycon - p.scale*ry
		tzSum += pt.zcon - p.scale*rz
	}
	p.tx = txSum / float64(n)
	p.ty = tySum / float64(n)
	p.tz = tzSum / float64(n)
}

// 求偏导数雅可比矩阵，返回 3行 x 7列
func formJacobianAI(p TransformParams, x, y, z float64) [][]float64 {
	AI := make([][]float64, 3)
	for i := range AI {
		AI[i] = make([]float64, 7)
	}

	// 第 1 列: 相对尺度 S 的偏导 (其实就是旋转后的坐标)
	AI[0][0] = p.m[0][0]*x + p.m[1][0]*y + p.m[2][0]*z
	AI[1][0] = p.m[0][1]*x + p.m[1][1]*y + p.m[2][1]*z
	AI[2][0] = p.m[0][2]*x + p.m[1][2]*y + p.m[2][2]*z

	// 第 2 列: 相对 Omega 的偏导
	AI[0][1] = 0.0
	AI[1][1] = -p.scale * AI[2][0]
	AI[2][1] = p.scale * AI[1][0]

	so, co := math.Sin(p.omega), math.Cos(p.omega)
	sp, cp := math.Sin(p.phi), math.Cos(p.phi)
	sk, ck := math.Sin(p.kappa), math.Cos(p.kappa)

	// 第 3 列: 相对 Phi 的偏导
	AI[0][2] = p.scale * (-sp*ck*x + sp*sk*y + cp*z)
	AI[1][2] = p.scale * (so*cp*ck*x - so*cp*sk*y + so*sp*z)
	AI[2][2] = p.scale * (-co*cp*ck*x + co*cp*sk*y - co*sp*z)

	// 第 4 列: 相对 Kappa 的偏导
	AI[0][3] = p.scale * (p.m[1][0]*x - p.m[0][0]*y)
	AI[1][3] = p.scale * (p.m[1][1]*x - p.m[0][1]*y)
	AI[2][3] = p.scale * (p.m[1][2]*x - p.m[0][2]*y)

	// 第 5-7 列: 相对平移量 Tx, Ty, Tz 的偏导
	AI[0][4], AI[1][5], AI[2][6] = 1.0, 1.0, 1.0

	return AI
}

func calcRotationMatrixOPK(p *TransformParams) {
	so, co := math.Sin(p.omega), math.Cos(p.omega)
	sp, cp := math.Sin(p.phi), math.Cos(p.phi)
	sk, ck := math.Sin(p.kappa), math.Cos(p.kappa)

	p.m[0][0] = cp * ck
	p.m[0][1] = so*sp*ck + co*sk
	p.m[0][2] = -co*sp*ck + so*sk
	p.m[1][0] = -cp * sk
	p.m[1][1] = -so*sp*sk + co*ck
	p.m[1][2] = co*sp*sk + so*ck
	p.m[2][0] = sp
	p.m[2][1] = -so * cp
	p.m[2][2] = co * cp
}

// 辅助数学与工具函数
func sq(v float64) float64 { return v * v }

func calcPlaneCoeffs(x1, y1, z1, x2, y2, z2, x3, y3, z3 float64) (A, B, C float64) {
	A = y1*z2 + y2*z3 + y3*z1 - y1*z3 - y2*z1 - y3*z2
	B = -(x1*z2 + x2*z3 + x3*z1 - x1*z3 - x2*z1 - x3*z2)
	C = x1*y2 + x2*y3 + x3*y1 - x1*y3 - x2*y1 - x3*y2
	return
}

func getRotMatTSA(tilt, swing, az float64) [][]float64 {
	st, ct := math.Sin(tilt), math.Cos(tilt)
	ss, cs := math.Sin(swing), math.Cos(swing)
	sa, ca := math.Sin(az), math.Cos(az)
	return [][]float64{
		{-ca*cs - sa*ct*ss, sa*cs - ca*ct*ss, -st * ss},
		{ca*ss - sa*ct*cs, -sa*ss - ca*ct*cs, -st * cs},
		{-sa * st, -ca * st, ct},
	}
}

func applyRot2D(m [][]float64, x, y, z float64) (rx, ry float64) {
	rx = m[0][0]*x + m[0][1]*y + m[0][2]*z
	ry = m[1][0]*x + m[1][1]*y + m[1][2]*z
	return
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

func inverseMat(A [][]float64) [][]float64 {
	n := len(A)
	mat := make([][]float64, n)
	inv := make([][]float64, n)
	for i := range mat {
		mat[i] = make([]float64, n)
		inv[i] = make([]float64, n)
		copy(mat[i], A[i])
		inv[i][i] = 1.0
	}
	for i := 0; i < n; i++ {
		max := i
		for k := i + 1; k < n; k++ {
			if math.Abs(mat[k][i]) > math.Abs(mat[max][i]) {
				max = k
			}
		}
		mat[i], mat[max] = mat[max], mat[i]
		inv[i], inv[max] = inv[max], inv[i]

		pivot := mat[i][i]
		for j := 0; j < n; j++ {
			mat[i][j] /= pivot
			inv[i][j] /= pivot
		}
		for j := 0; j < n; j++ {
			if i != j {
				f := mat[j][i]
				for k := 0; k < n; k++ {
					mat[j][k] -= f * mat[i][k]
					inv[j][k] -= f * inv[i][k]
				}
			}
		}
	}
	return inv
}

func loadAbsOrData() ([]CommonPoint, []UnknownPoint) {
	data := `
a   390.35  499.63  469.43  607.54  501.63  469.09
b   371.68  630.84   81.25  589.98  632.36   82.81 
c   425.65  419.07   82.49  643.65  421.28   83.50
d   410.50  438.31   81.13  628.58  440.51   82.27 
e   448.22  295.83   97.79  666.27  298.16   98.29 
f   414.60  709.39  101.77  632.59  710.62  103.01
#
1   611.37  498.98  470.45
2   637.49  323.67  85.67
3   573.32  401.51  84.48
4   647.00  373.97  83.76
5   533.51  285.01  87.13
`
	lines := strings.Split(strings.TrimSpace(data), "\n")
	var comPts []CommonPoint
	var unkPts []UnknownPoint
	isUnknown := false

	for _, line := range lines {
		line = strings.TrimSpace(line)
		if line == "" {
			continue
		}
		if line == "#" {
			isUnknown = true
			continue
		}

		if !isUnknown {
			var p CommonPoint
			fmt.Sscanf(line, "%s %f %f %f %f %f %f", &p.name, &p.xarb, &p.yarb, &p.zarb, &p.xcon, &p.ycon, &p.zcon)
			comPts = append(comPts, p)
		} else {
			var p UnknownPoint
			fmt.Sscanf(line, "%s %f %f %f", &p.name, &p.x, &p.y, &p.z)
			unkPts = append(unkPts, p)
		}
	}
	return comPts, unkPts
}
