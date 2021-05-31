package main

import (
	"fmt"
	"os"

	dem "github.com/markus-wa/demoinfocs-golang/v2/pkg/demoinfocs"
	events "github.com/markus-wa/demoinfocs-golang/v2/pkg/demoinfocs/events"
)

// This contains docs on the parser https://pkg.go.dev/github.com/markus-wa/demoinfocs-golang/v2/pkg/demoinfocs#example-Parser
// This is all the events https://pkg.go.dev/github.com/markus-wa/demoinfocs-golang/v2@v2.8.1/pkg/demoinfocs/events

func main() {
	f, err := os.Open("E:/Projects/GRAIL_PCGML_tmaurer_summer_2021/csgo/logs/demos/1-0a3a6708-2074-4d69-906f-fe6bd54c644c.dem")
	if err != nil {
		panic(err)
	}
	defer f.Close()

	p := dem.NewParser(f)
	defer p.Close()

	count := 0

	// Register handler on frames
	p.RegisterEventHandler(func(e events.FrameDone) {
		count += 1
		// fmt.Println(p.CurrentTime()) -> can get the current time
		// fmt.Println(p.GameState().Participants().Playing()[0].LastAlivePosition) -> Can get player positions
	})

	// Register handler on shots fired
	p.RegisterEventHandler(func(e events.WeaponFire) {
		//fmt.Println(e.Weapon.Type) -> Can get shots fired
	})

	// Parse to end
	err = p.ParseToEnd()
	if err != nil {
		panic(err)
	}

	fmt.Println(count)
}
