package main

import (
	"fmt"
	"os"
	"time"

	dem "github.com/markus-wa/demoinfocs-golang/v2/pkg/demoinfocs"
	events "github.com/markus-wa/demoinfocs-golang/v2/pkg/demoinfocs/events"
)

// This contains docs on the parser https://pkg.go.dev/github.com/markus-wa/demoinfocs-golang/v2/pkg/demoinfocs#example-Parser
// This is all the events https://pkg.go.dev/github.com/markus-wa/demoinfocs-golang/v2@v2.8.1/pkg/demoinfocs/events

func generatePlayerMap(state dem.GameState) map[string]int {
	var players map[string]int
	for i := 0; i < len(state.Participants().Playing()); i++ {
		players[state.Participants().Playing()[i].Name] = i
	}

	return players
}

func main() {
	f, err := os.Open("E:/Projects/GRAIL_PCGML_tmaurer_summer_2021/csgo/logs/demos/1-0ad04954-9047-41d7-8a2b-1898ac25de65.dem")
	if err != nil {
		panic(err)
	}
	defer f.Close()

	p := dem.NewParser(f)
	defer p.Close()

	// 10 Players
	// 2700 = 45 minutes * 60 seconds
	// timestamp vector = PosX, PosY, PosZ, VelocityX, VelocityY, VelocityZ, ViewX, ViewY, Health, Weapon1, Weapon2, Crouched, Jumped,
	// Alive, Shooting, Flash, Grenade, Incendiary, Smoke, Decoy, Planted Bomb, Defused Bomb, Blind, Reloaded, Scoped = 25 items
	// Maybe include buy info for the time when players are buying?
	// TODO: demoVector := [10][2700][25]float64{}

	// Find the start of the match
	for !p.GameState().IsMatchStarted() {
		p.ParseNextFrame()
	}

	// TODO: playerMap := generatePlayerMap(p.GameState())

	// TODO: Register handler on each type of event
	p.RegisterEventHandler(func(e events.WeaponFire) {
		fmt.Println(e.Weapon.Type)
	})

	// PARSE TO 45 Minutes
	startTime := p.CurrentTime()
out:
	for second := 0; second < 2700; second++ {
		for p.CurrentTime()-startTime < time.Second {
			EoF, err := p.ParseNextFrame()
			if err != nil {
				panic(err)
			}
			if !EoF {
				break out
			}
		}
		fmt.Println(second)

		startTime = p.CurrentTime()
	}

}
