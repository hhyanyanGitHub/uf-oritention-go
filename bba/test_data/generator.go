// 文件名: generator.go
// 运行方法: go run generator.go
package main

import (
	"encoding/csv"
	"encoding/json"
	"fmt"
	"math"
	"os"
	"path/filepath"
)

// 把结构体提到全局作用域
type Pt struct {
	id      int
	name    string
	x, y, z float64
	isFixed bool
}

type Cam struct {
	id         int
	xl, yl, zl float64
	o, p, k, f float64
}

type Obs struct {
	camID, ptID int
	x, y        float64
}

func main() {
	outDir := "dataset_30cams"
	os.MkdirAll(outDir, os.ModePerm)

	fmt.Println("开始生成 5x6 (30张影像) 无人机航测大型仿真数据集...")

	// 1. 生成地面点 (起伏地形: Z = 20*sin(X/100) + 30*cos(Y/100) + 50)
	// X: 0~500, Y: 0~400, 步长 50，共 11 * 9 = 99 个点
	var truePoints []Pt
	ptID := 0
	for y := 0.0; y <= 400; y += 50 {
		for x := 0.0; x <= 500; x += 50 {
			z := 20*math.Sin(x/100.0) + 30*math.Cos(y/100.0) + 50
			// 选 5 个点作为控制点 (四角 + 中心)
			isFixed := (x == 0 && y == 0) || (x == 500 && y == 0) || (x == 0 && y == 400) || (x == 500 && y == 400) || (x == 250 && y == 200)
			name := fmt.Sprintf("Pt_%d", ptID)
			if isFixed {
				name = "Ctrl_" + name
			}
			truePoints = append(truePoints, Pt{ptID, name, x, y, z, isFixed})
			ptID++
		}
	}

	// 2. 生成相机 (5条航线，每条6张，高度 600m)
	var trueCams []Cam
	camID := 0
	focal := 50.0                      // 50mm 镜头
	for y := 0.0; y <= 400; y += 100 { // 5 条航线
		for x := 0.0; x <= 500; x += 100 { // 没条航线 6 张
			trueCams = append(trueCams, Cam{camID, x, y, 600, 0.01, -0.01, 0.005, focal})
			camID++
		}
	}

	// 3. 物理仿真：生成观测值 (模拟 36mm x 24mm 传感器视场)
	var observations []Obs
	for _, c := range trueCams {
		for _, p := range truePoints {
			x, y := projectExact(c, p)
			// 如果该点在相机的物理传感器范围内，则视为“被拍到”
			if math.Abs(x) <= 18.0 && math.Abs(y) <= 12.0 {
				observations = append(observations, Obs{c.id, p.id, x, y})
			}
		}
	}
	fmt.Printf("物理仿真完毕: 生成 %d 个相机, %d 个地面点, 产生 %d 条交叉光线观测.\n", len(trueCams), len(truePoints), len(observations))

	// 4. 写入文件 (故意给初始值添加误差，留给 BBA 去平差)
	writeCameras(filepath.Join(outDir, "cameras.csv"), trueCams)
	writePoints(filepath.Join(outDir, "points.csv"), truePoints)
	writeObservations(filepath.Join(outDir, "observations.csv"), observations)

	// 5. 写入主工程配置 JSON
	config := map[string]string{
		"project_name": "Mini_City_Block_30",
		"camera_file":  "cameras.csv",
		"point_file":   "points.csv",
		"obs_file":     "observations.csv",
	}
	b, _ := json.MarshalIndent(config, "", "  ")
	os.WriteFile(filepath.Join(outDir, "project.json"), b, 0644)
	fmt.Printf("-> 数据集已生成至目录: ./%s/\n", outDir)
}

// ---- 底层仿真投影算法 ----
func projectExact(c Cam, p Pt) (float64, float64) {
	dx, dy, dz := p.x-c.xl, p.y-c.yl, p.z-c.zl
	so, co := math.Sin(c.o), math.Cos(c.o)
	sp, cp := math.Sin(c.p), math.Cos(c.p)
	sk, ck := math.Sin(c.k), math.Cos(c.k)
	m11, m12, m13 := cp*ck, so*sp*ck+co*sk, -co*sp*ck+so*sk
	m21, m22, m23 := -cp*sk, -so*sp*sk+co*ck, co*sp*sk+so*ck
	m31, m32, m33 := sp, -so*cp, co*cp

	r := m11*dx + m12*dy + m13*dz
	s := m21*dx + m22*dy + m23*dz
	q := m31*dx + m32*dy + m33*dz
	return -c.f * r / q, -c.f * s / q
}

// 写入相机 (故意加 15 米、10 米的巨大初始误差)
func writeCameras(path string, cams []Cam) {
	f, _ := os.Create(path)
	defer f.Close()
	w := csv.NewWriter(f)
	w.Write([]string{"id", "f", "xl", "yl", "zl", "omega", "phi", "kappa"})
	for _, c := range cams {
		w.Write([]string{
			fmt.Sprint(c.id), fmt.Sprint(c.f),
			fmt.Sprintf("%.3f", c.xl+15.0), fmt.Sprintf("%.3f", c.yl-10.0), fmt.Sprintf("%.3f", c.zl+25.0), // 初始位置加误差
			"0.0", "0.0", "0.0", // 初始角度全部猜 0
		})
	}
	w.Flush()
}

// 写入点 (未知地物点全部猜 Z=0)
func writePoints(path string, pts []Pt) {
	f, _ := os.Create(path)
	defer f.Close()
	w := csv.NewWriter(f)
	w.Write([]string{"id", "name", "X", "Y", "Z", "isFixed"})
	for _, p := range pts {
		z := p.z
		if !p.isFixed {
			z = 0.0 // 破坏未知点的高程真值
		}
		fixed := "0"
		if p.isFixed {
			fixed = "1"
		}
		w.Write([]string{fmt.Sprint(p.id), p.name, fmt.Sprintf("%.3f", p.x), fmt.Sprintf("%.3f", p.y), fmt.Sprintf("%.3f", z), fixed})
	}
	w.Flush()
}

// 写入精确观测值
func writeObservations(path string, obs []Obs) {
	f, _ := os.Create(path)
	defer f.Close()
	w := csv.NewWriter(f)
	w.Write([]string{"camID", "ptID", "x", "y"})
	for _, o := range obs {
		w.Write([]string{fmt.Sprint(o.camID), fmt.Sprint(o.ptID), fmt.Sprintf("%.6f", o.x), fmt.Sprintf("%.6f", o.y)})
	}
	w.Flush()
}
