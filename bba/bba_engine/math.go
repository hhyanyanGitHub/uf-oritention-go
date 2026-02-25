// 文件名: math.go
package main

import "math"

func NewMat(r, c int) [][]float64 {
	m := make([][]float64, r)
	for i := range m {
		m[i] = make([]float64, c)
	}
	return m
}

func Transpose(a [][]float64) [][]float64 {
	out := NewMat(len(a[0]), len(a))
	for i := range a {
		for j := range a[0] {
			out[j][i] = a[i][j]
		}
	}
	return out
}

func MultiplyMat(a, b [][]float64) [][]float64 {
	out := NewMat(len(a), len(b[0]))
	for i := range out {
		for j := range out[0] {
			for x := range a[0] {
				out[i][j] += a[i][x] * b[x][j]
			}
		}
	}
	return out
}

func MultiplyMatVec(a [][]float64, v []float64) []float64 {
	out := make([]float64, len(a))
	for i := range a {
		for x := range a[0] {
			out[i] += a[i][x] * v[x]
		}
	}
	return out
}

func InvertBlockDiagonal(mat [][]float64, bs int) [][]float64 {
	n := len(mat)
	inv := NewMat(n, n)
	for i := 0; i < n; i += bs {
		b := NewMat(bs, bs)
		for r := 0; r < bs; r++ {
			for c := 0; c < bs; c++ {
				b[r][c] = mat[i+r][i+c]
			}
		}
		binv := inverse3x3(b)
		for r := 0; r < bs; r++ {
			for c := 0; c < bs; c++ {
				inv[i+r][i+c] = binv[r][c]
			}
		}
	}
	return inv
}

func inverse3x3(m [][]float64) [][]float64 {
	det := m[0][0]*(m[1][1]*m[2][2]-m[1][2]*m[2][1]) - m[0][1]*(m[1][0]*m[2][2]-m[1][2]*m[2][0]) + m[0][2]*(m[1][0]*m[2][1]-m[1][1]*m[2][0])
	inv := NewMat(3, 3)
	inv[0][0] = (m[1][1]*m[2][2] - m[1][2]*m[2][1]) / det
	inv[0][1] = (m[0][2]*m[2][1] - m[0][1]*m[2][2]) / det
	inv[0][2] = (m[0][1]*m[1][2] - m[0][2]*m[1][1]) / det
	inv[1][0] = (m[1][2]*m[2][0] - m[1][0]*m[2][2]) / det
	inv[1][1] = (m[0][0]*m[2][2] - m[0][2]*m[2][0]) / det
	inv[1][2] = (m[0][2]*m[1][0] - m[0][0]*m[1][2]) / det
	inv[2][0] = (m[1][0]*m[2][1] - m[1][1]*m[2][0]) / det
	inv[2][1] = (m[0][1]*m[2][0] - m[0][0]*m[2][1]) / det
	inv[2][2] = (m[0][0]*m[1][1] - m[0][1]*m[1][0]) / det
	return inv
}

func SolveGaussian(A [][]float64, B []float64) []float64 {
	n := len(B)
	mat := NewMat(n, n+1)
	for i := 0; i < n; i++ {
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

// InverseMat 使用高斯-若尔当消元法求解矩阵的逆 (用于协方差阵计算)
func InverseMat(A [][]float64) [][]float64 {
	n := len(A)
	mat := NewMat(n, n)
	inv := NewMat(n, n)
	for i := range mat {
		copy(mat[i], A[i])
		inv[i][i] = 1.0 // 初始化为单位阵
	}

	for i := 0; i < n; i++ {
		// 主元选取
		max := i
		for k := i + 1; k < n; k++ {
			if math.Abs(mat[k][i]) > math.Abs(mat[max][i]) {
				max = k
			}
		}
		mat[i], mat[max] = mat[max], mat[i]
		inv[i], inv[max] = inv[max], inv[i]

		pivot := mat[i][i]
		if math.Abs(pivot) < 1e-12 {
			continue
		} // 防止除零(理论上满秩矩阵不会发生)

		for j := 0; j < n; j++ {
			mat[i][j] /= pivot
			inv[i][j] /= pivot
		}
		for j := 0; j < n; j++ {
			if i != j {
				factor := mat[j][i]
				for k := 0; k < n; k++ {
					mat[j][k] -= factor * mat[i][k]
					inv[j][k] -= factor * inv[i][k]
				}
			}
		}
	}
	return inv
}
