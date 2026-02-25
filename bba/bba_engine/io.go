// 文件名: io.go
package main

import (
	"encoding/csv"
	"encoding/json"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"strconv"
)

type Camera struct {
	ID                int
	Omega, Phi, Kappa float64
	XL, YL, ZL        float64
	F                 float64
	M                 [3][3]float64
	// 新增：精度评估 (Standard Deviations)
	SOmega, SPhi, SKappa float64
	SXL, SYL, SZL        float64
}

type GroundPoint struct {
	ID      int
	Name    string
	X, Y, Z float64
	IsFixed bool
	// 新增：精度评估
	SX, SY, SZ float64
}

type Observation struct {
	CamID, PtID int
	X, Y        float64
}

// LoadProject (保持原样)
func LoadProject(jsonPath string) ([]*Camera, []*GroundPoint, []Observation, error) {
	b, err := os.ReadFile(jsonPath)
	if err != nil {
		return nil, nil, nil, err
	}
	var config map[string]string
	json.Unmarshal(b, &config)
	baseDir := filepath.Dir(jsonPath)
	return loadCameras(filepath.Join(baseDir, config["camera_file"])),
		loadPoints(filepath.Join(baseDir, config["point_file"])),
		loadObservations(filepath.Join(baseDir, config["obs_file"])), nil
}

func loadCameras(path string) []*Camera {
	f, _ := os.Open(path)
	defer f.Close()
	lines, _ := csv.NewReader(f).ReadAll()
	var cams []*Camera
	for i, l := range lines {
		if i == 0 {
			continue
		}
		id, _ := strconv.Atoi(l[0])
		focal, _ := strconv.ParseFloat(l[1], 64)
		xl, _ := strconv.ParseFloat(l[2], 64)
		yl, _ := strconv.ParseFloat(l[3], 64)
		zl, _ := strconv.ParseFloat(l[4], 64)
		o, _ := strconv.ParseFloat(l[5], 64)
		p, _ := strconv.ParseFloat(l[6], 64)
		k, _ := strconv.ParseFloat(l[7], 64)
		cams = append(cams, &Camera{ID: id, F: focal, XL: xl, YL: yl, ZL: zl, Omega: o, Phi: p, Kappa: k})
	}
	return cams
}

func loadPoints(path string) []*GroundPoint {
	f, _ := os.Open(path)
	defer f.Close()
	lines, _ := csv.NewReader(f).ReadAll()
	var pts []*GroundPoint
	for i, l := range lines {
		if i == 0 {
			continue
		}
		id, _ := strconv.Atoi(l[0])
		x, _ := strconv.ParseFloat(l[2], 64)
		y, _ := strconv.ParseFloat(l[3], 64)
		z, _ := strconv.ParseFloat(l[4], 64)
		fixed := l[5] == "1"
		pts = append(pts, &GroundPoint{ID: id, Name: l[1], X: x, Y: y, Z: z, IsFixed: fixed})
	}
	return pts
}

func loadObservations(path string) []Observation {
	f, _ := os.Open(path)
	defer f.Close()
	lines, _ := csv.NewReader(f).ReadAll()
	var obs []Observation
	for i, l := range lines {
		if i == 0 {
			continue
		}
		cid, _ := strconv.Atoi(l[0])
		pid, _ := strconv.Atoi(l[1])
		x, _ := strconv.ParseFloat(l[2], 64)
		y, _ := strconv.ParseFloat(l[3], 64)
		obs = append(obs, Observation{CamID: cid, PtID: pid, X: x, Y: y})
	}
	return obs
}

// ================= 新增：工程结果输出模块 =================

// ExportReport 导出平差报告和结果 CSV
func ExportReport(projPath string, cams []*Camera, pts []*GroundPoint, s0 float64) {
	baseDir := filepath.Dir(projPath)

	// 1. 生成可读性极强的文本报告
	reportPath := filepath.Join(baseDir, "Adjustment_Report.txt")
	f, _ := os.Create(reportPath)
	defer f.Close()

	head := fmt.Sprintf("====================================================\n")
	head += fmt.Sprintf("          光束法区域网平差 终极精度报告\n")
	head += fmt.Sprintf("====================================================\n")
	head += fmt.Sprintf("【全局指标】\n")
	head += fmt.Sprintf("单位权中误差 (Sigma 0) : %.6f mm\n", s0)
	head += fmt.Sprintf("参与平差相机数 : %d\n", len(cams))
	head += fmt.Sprintf("参与平差地物点 : %d\n", len(pts))
	head += fmt.Sprintf("----------------------------------------------------\n\n")

	f.WriteString(head)
	fmt.Print(head)

	f.WriteString("【相机平差结果与精度 (X,Y,Z单位:m, 姿态单位:度)】\n")
	f.WriteString(fmt.Sprintf("%-5s %10s %10s %10s | %8s %8s %8s | %7s %7s %7s\n", "CamID", "XL", "YL", "ZL", "Omega", "Phi", "Kappa", "SD_X", "SD_Y", "SD_Z"))
	for _, c := range cams {
		line := fmt.Sprintf("Cam%02d %10.4f %10.4f %10.4f | %8.4f %8.4f %8.4f | %7.4f %7.4f %7.4f\n",
			c.ID, c.XL, c.YL, c.ZL, c.Omega*180/math.Pi, c.Phi*180/math.Pi, c.Kappa*180/math.Pi, c.SXL, c.SYL, c.SZL)
		f.WriteString(line)
	}

	f.WriteString("\n【地面点三维坐标与精度 (单位:m)】\n")
	f.WriteString(fmt.Sprintf("%-8s %10s %10s %10s | %7s %7s %7s\n", "PtName", "X", "Y", "Z", "SD_X", "SD_Y", "SD_Z"))
	for _, p := range pts {
		if p.IsFixed {
			f.WriteString(fmt.Sprintf("%-8s %10.4f %10.4f %10.4f | %7s %7s %7s (固定控制点)\n", p.Name, p.X, p.Y, p.Z, "-", "-", "-"))
		} else {
			f.WriteString(fmt.Sprintf("%-8s %10.4f %10.4f %10.4f | %7.4f %7.4f %7.4f\n", p.Name, p.X, p.Y, p.Z, p.SX, p.SY, p.SZ))
		}
	}

	fmt.Printf(">>> 详细平差报告已导出至: %s\n", reportPath)

	// 2. 导出 GIS 软件可直接读取的 CSV 点云文件
	csvPath := filepath.Join(baseDir, "Adjusted_Points.csv")
	fCsv, _ := os.Create(csvPath)
	defer fCsv.Close()
	w := csv.NewWriter(fCsv)
	w.Write([]string{"Name", "X", "Y", "Z", "SD_X", "SD_Y", "SD_Z", "IsControl"})
	for _, p := range pts {
		ctrl := "0"
		if p.IsFixed {
			ctrl = "1"
		}
		w.Write([]string{
			p.Name, fmt.Sprintf("%.4f", p.X), fmt.Sprintf("%.4f", p.Y), fmt.Sprintf("%.4f", p.Z),
			fmt.Sprintf("%.4f", p.SX), fmt.Sprintf("%.4f", p.SY), fmt.Sprintf("%.4f", p.SZ), ctrl,
		})
	}
	w.Flush()
	fmt.Printf(">>> 最终三维点云已导出至: %s\n", csvPath)
}
