// 文件名: main.go
package main

import (
	"flag"
	"fmt"
	"time"
)

func main() {
	projPath := flag.String("proj", "", "Path to the project.json file")
	flag.Parse()

	if *projPath == "" {
		fmt.Println("Error: 必须提供工程文件。用法: go run . -proj ./dataset_30cams/project.json")
		return
	}

	fmt.Printf(">>> 启动工业级 BBA 求解引擎 <<<\n")
	fmt.Printf("加载工程: %s\n", *projPath)

	cameras, points, obs, err := LoadProject(*projPath)
	if err != nil {
		fmt.Printf("加载失败: %v\n", err)
		return
	}

	fmt.Printf("成功载入: 相机 %d 台, 测点 %d 个, 观测光线 %d 条\n", len(cameras), len(points), len(obs))

	start := time.Now()
	// 假设您已经将 solver.go 改为返回 s0
	final_s0 := RunBundleAdjustment(cameras, points, obs, 15, 1e-6)
	elapsed := time.Since(start)

	fmt.Printf(">>> 平差解算完毕! 核心耗时: %v <<<\n\n", elapsed)

	// 调用结果导出器！
	ExportReport(*projPath, cameras, points, final_s0)
}
