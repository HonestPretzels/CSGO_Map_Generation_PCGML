package main

import (
	"fmt"

	"github.com/kshedden/gonpy"
)

func Flatten(arr [][][]float64) []float64 {
	out := make([]float64, len(arr)*len(arr[0])*len(arr[0][0]))
	width := len(arr[0])
	depth := len(arr[0][0])
	for i := range arr {
		for j := range arr[i] {
			for k := range arr[i][j] {
				idx := ((i*width + j) * depth) + k
				out[idx] = arr[i][j][k]
			}
		}
	}
	return out
}

func main() {
	data := make([][][]float64, 4)
	for i := range data {
		data[i] = make([][]float64, 3)
		for j := range data[i] {
			data[i][j] = make([]float64, 2)
			for k := range data[i][j] {
				data[i][j][k] = float64(i+j+k) / 10
			}
		}
	}

	fmt.Println(data)
	fmt.Println(Flatten(data))
	w, _ := gonpy.NewFileWriter("data.npy")
	w.Shape = []int{4, 3, 2}
	w.Version = 2
	_ = w.WriteFloat64(Flatten(data))
}
