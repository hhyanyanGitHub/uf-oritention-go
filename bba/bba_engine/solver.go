// 文件名: solver.go
package main

import (
	"fmt"
	"math"
)

// RunBundleAdjustment 执行舒尔补平差，并返回最终的单位权中误差 s0
func RunBundleAdjustment(cams []*Camera, pts []*GroundPoint, obs []Observation, maxIter int, tol float64) float64 {
	for _, c := range cams {
		UpdateRotation(c)
	}

	camIdxMap := make(map[int]int)
	for i, c := range cams {
		camIdxMap[c.ID] = i
	}

	unkPtIdxMap := make(map[int]int)
	unkCount := 0
	for _, p := range pts {
		if !p.IsFixed {
			unkPtIdxMap[p.ID] = unkCount
			unkCount++
		}
	}

	numCams := len(cams)
	var s0, s0old float64 = 1.0e30, 0

	for iter := 1; iter <= maxIter; iter++ {
		s0old = s0
		N11 := NewMat(6*numCams, 6*numCams)
		N22 := NewMat(3*unkCount, 3*unkCount)
		N12 := NewMat(6*numCams, 3*unkCount)
		U1 := make([]float64, 6*numCams)
		U2 := make([]float64, 3*unkCount)

		sumResSq := 0.0

		for _, o := range obs {
			cIdx := camIdxMap[o.CamID]
			cam := cams[cIdx]

			var pt *GroundPoint
			for _, p := range pts {
				if p.ID == o.PtID {
					pt = p
					break
				}
			}

			pIdx, isUnk := unkPtIdxMap[pt.ID]

			epsX, epsY, Ac, Ap := CalcPartials(cam, pt, o.X, o.Y)
			sumResSq += epsX*epsX + epsY*epsY

			rowC := cIdx * 6
			for r := 0; r < 6; r++ {
				U1[rowC+r] += Ac[0][r]*epsX + Ac[1][r]*epsY
				for c := 0; c < 6; c++ {
					N11[rowC+r][rowC+c] += Ac[0][r]*Ac[0][c] + Ac[1][r]*Ac[1][c]
				}
			}

			if isUnk {
				rowP := pIdx * 3
				for r := 0; r < 3; r++ {
					U2[rowP+r] += Ap[0][r]*epsX + Ap[1][r]*epsY
					for c := 0; c < 3; c++ {
						N22[rowP+r][rowP+c] += Ap[0][r]*Ap[0][c] + Ap[1][r]*Ap[1][c]
					}
				}
				for r := 0; r < 6; r++ {
					for c := 0; c < 3; c++ {
						N12[rowC+r][rowP+c] += Ac[0][r]*Ap[0][c] + Ac[1][r]*Ap[1][c]
					}
				}
			}
		}

		s0 = math.Sqrt(sumResSq / float64(len(obs)*2-(6*numCams+3*unkCount)))
		fmt.Printf(" [Iter %2d] 中误差 S0 = %.6f\n", iter, s0)

		// -------------------------------------------------------------
		// 舒尔补矩阵组装 (必须在精度评估之前计算！)
		// -------------------------------------------------------------
		N22_inv := InvertBlockDiagonal(N22, 3)
		N12_x_N22inv := MultiplyMat(N12, N22_inv)
		N12T := Transpose(N12)
		SchurTermN := MultiplyMat(N12_x_N22inv, N12T)
		SchurTermU := MultiplyMatVec(N12_x_N22inv, U2)

		N_reduced := NewMat(6*numCams, 6*numCams)
		U_reduced := make([]float64, 6*numCams)
		for r := 0; r < 6*numCams; r++ {
			U_reduced[r] = U1[r] - SchurTermU[r]
			for c := 0; c < 6*numCams; c++ {
				N_reduced[r][c] = N11[r][c] - SchurTermN[r][c]
			}
		}

		// =============================================================
		// 收敛判断与极其硬核的精度传播定律评估！
		// =============================================================
		if math.Abs(s0old-s0)/s0 < tol {

			// 1. 相机精度：Q_cam = (N_reduced)^-1
			Q_cam := InverseMat(N_reduced)
			for i := 0; i < numCams; i++ {
				c := cams[i]
				c.SOmega = s0 * math.Sqrt(math.Abs(Q_cam[i*6+0][i*6+0]))
				c.SPhi = s0 * math.Sqrt(math.Abs(Q_cam[i*6+1][i*6+1]))
				c.SKappa = s0 * math.Sqrt(math.Abs(Q_cam[i*6+2][i*6+2]))
				c.SXL = s0 * math.Sqrt(math.Abs(Q_cam[i*6+3][i*6+3]))
				c.SYL = s0 * math.Sqrt(math.Abs(Q_cam[i*6+4][i*6+4]))
				c.SZL = s0 * math.Sqrt(math.Abs(Q_cam[i*6+5][i*6+5]))
			}

			// 2. 地面点精度传播：Q_pt = N22^-1 + M^T * Q_cam * M (其中 M = N12 * N22^-1)
			M := N12_x_N22inv
			MT := Transpose(M)
			MT_Qcam := MultiplyMat(MT, Q_cam)
			Q_pt_part2 := MultiplyMat(MT_Qcam, M) // 这是一个 3n x 3n 的矩阵

			for pIdx := 0; pIdx < unkCount; pIdx++ {
				base := pIdx * 3
				varX := N22_inv[base+0][base+0] + Q_pt_part2[base+0][base+0]
				varY := N22_inv[base+1][base+1] + Q_pt_part2[base+1][base+1]
				varZ := N22_inv[base+2][base+2] + Q_pt_part2[base+2][base+2]

				for _, p := range pts {
					if idx, ok := unkPtIdxMap[p.ID]; ok && idx == pIdx {
						p.SX = s0 * math.Sqrt(math.Abs(varX))
						p.SY = s0 * math.Sqrt(math.Abs(varY))
						p.SZ = s0 * math.Sqrt(math.Abs(varZ))
					}
				}
			}
			break // 计算完精度后，直接退出迭代！
		}

		// 如果没收敛，继续解算并应用改正数
		deltaCam := SolveGaussian(N_reduced, U_reduced)

		for i := 0; i < numCams; i++ {
			c := cams[i]
			c.Omega += deltaCam[i*6+0]
			c.Phi += deltaCam[i*6+1]
			c.Kappa += deltaCam[i*6+2]
			c.XL += deltaCam[i*6+3]
			c.YL += deltaCam[i*6+4]
			c.ZL += deltaCam[i*6+5]
			UpdateRotation(c)
		}

		N12T_x_dCam := MultiplyMatVec(N12T, deltaCam)
		U2_minus := make([]float64, 3*unkCount)
		for i := range U2 {
			U2_minus[i] = U2[i] - N12T_x_dCam[i]
		}
		deltaPt := MultiplyMatVec(N22_inv, U2_minus)

		for _, p := range pts {
			if idx, ok := unkPtIdxMap[p.ID]; ok {
				p.X += deltaPt[idx*3+0]
				p.Y += deltaPt[idx*3+1]
				p.Z += deltaPt[idx*3+2]
			}
		}
	}
	return s0
}

// =========================================================
// 底层共线方程算子
// =========================================================
func CalcPartials(cam *Camera, pt *GroundPoint, obs_x, obs_y float64) (float64, float64, [2][6]float64, [2][3]float64) {
	dx, dy, dz := pt.X-cam.XL, pt.Y-cam.YL, pt.Z-cam.ZL
	r := cam.M[0][0]*dx + cam.M[0][1]*dy + cam.M[0][2]*dz
	s := cam.M[1][0]*dx + cam.M[1][1]*dy + cam.M[1][2]*dz
	q := cam.M[2][0]*dx + cam.M[2][1]*dy + cam.M[2][2]*dz

	epsX := obs_x - (-cam.F * r / q)
	epsY := obs_y - (-cam.F * s / q)

	f_q2 := cam.F / (q * q)
	so, co := math.Sin(cam.Omega), math.Cos(cam.Omega)
	sp, cp := math.Sin(cam.Phi), math.Cos(cam.Phi)
	sk, ck := math.Sin(cam.Kappa), math.Cos(cam.Kappa)

	var Ac [2][6]float64
	Ac[0][0] = f_q2 * (r*(-cam.M[2][2]*dy+cam.M[2][1]*dz) - q*(-cam.M[0][2]*dy+cam.M[0][1]*dz))
	Ac[0][1] = f_q2 * (r*(cp*dx+so*sp*dy-co*sp*dz) - q*(-sp*ck*dx+so*cp*ck*dy-co*cp*ck*dz))
	Ac[0][2] = -cam.F * s / q
	Ac[0][3] = -cam.F * (r*cam.M[2][0] - q*cam.M[0][0]) / (q * q)
	Ac[0][4] = -cam.F * (r*cam.M[2][1] - q*cam.M[0][1]) / (q * q)
	Ac[0][5] = -cam.F * (r*cam.M[2][2] - q*cam.M[0][2]) / (q * q)

	Ac[1][0] = f_q2 * (s*(-cam.M[2][2]*dy+cam.M[2][1]*dz) - q*(-cam.M[1][2]*dy+cam.M[1][1]*dz))
	Ac[1][1] = f_q2 * (s*(cp*dx+so*sp*dy-co*sp*dz) - q*(sp*sk*dx-so*cp*sk*dy+co*cp*sk*dz))
	Ac[1][2] = cam.F * r / q
	Ac[1][3] = -cam.F * (s*cam.M[2][0] - q*cam.M[1][0]) / (q * q)
	Ac[1][4] = -cam.F * (s*cam.M[2][1] - q*cam.M[1][1]) / (q * q)
	Ac[1][5] = -cam.F * (s*cam.M[2][2] - q*cam.M[1][2]) / (q * q)

	var Ap [2][3]float64
	Ap[0][0], Ap[0][1], Ap[0][2] = -Ac[0][3], -Ac[0][4], -Ac[0][5]
	Ap[1][0], Ap[1][1], Ap[1][2] = -Ac[1][3], -Ac[1][4], -Ac[1][5]
	return epsX, epsY, Ac, Ap
}

func UpdateRotation(c *Camera) {
	so, co := math.Sin(c.Omega), math.Cos(c.Omega)
	sp, cp := math.Sin(c.Phi), math.Cos(c.Phi)
	sk, ck := math.Sin(c.Kappa), math.Cos(c.Kappa)
	c.M[0][0], c.M[0][1], c.M[0][2] = cp*ck, so*sp*ck+co*sk, -co*sp*ck+so*sk
	c.M[1][0], c.M[1][1], c.M[1][2] = -cp*sk, -so*sp*sk+co*ck, co*sp*sk+so*ck
	c.M[2][0], c.M[2][1], c.M[2][2] = sp, -so*cp, co*cp
}
