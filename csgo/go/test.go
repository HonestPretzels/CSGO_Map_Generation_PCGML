package main

import (
	"fmt"
	"os"
	"time"

	dem "github.com/markus-wa/demoinfocs-golang/v2/pkg/demoinfocs"
	"github.com/markus-wa/demoinfocs-golang/v2/pkg/demoinfocs/common"
	"github.com/markus-wa/demoinfocs-golang/v2/pkg/demoinfocs/events"
)

// enum used to index different aspects of state for a player at a given time
type playerState int

const (
	PosX playerState = iota
	PosY
	PosZ
	VelocityX
	VelocityY
	VelocityZ
	ViewX
	ViewY
	Health
	ActiveWeapon
	Blind
	Scoped
	Reloaded
	Crouched
	Jumped
	Shooting
	Flash
	Grenade
	Incendiary
	Smoke
	Decoy
	PlantedBomb
)

// This contains docs on the parser https://pkg.go.dev/github.com/markus-wa/demoinfocs-golang/v2/pkg/demoinfocs#example-Parser
// This is all the events https://pkg.go.dev/github.com/markus-wa/demoinfocs-golang/v2@v2.8.1/pkg/demoinfocs/events

func generatePlayerMap(state dem.GameState) map[string]int {
	var players map[string]int
	players = make(map[string]int)
	ct := state.Participants().TeamMembers(state.TeamCounterTerrorists().Team())
	t := state.Participants().TeamMembers(state.TeamTerrorists().Team())
	for i := 0; i < len(ct); i++ {
		players[ct[i].Name] = i
	}
	for i := 0; i < len(t); i++ {
		players[t[i].Name] = 5 + i
	}

	return players
}

func boolToFloat(x bool) float64 {
	if x {
		return float64(1)
	}
	return float64(0)
}

func updatePlayerState(player *common.Player, second int, idx int, demoVector *[10][2700][23]float64) {
	// Postion
	demoVector[idx][second][PosX] = player.Position().X
	demoVector[idx][second][PosY] = player.Position().Y
	demoVector[idx][second][PosZ] = player.Position().Z
	// Velocity
	demoVector[idx][second][VelocityX] = player.Velocity().X
	demoVector[idx][second][VelocityY] = player.Velocity().Y
	demoVector[idx][second][VelocityZ] = player.Velocity().Z
	// View
	demoVector[idx][second][ViewX] = float64(player.ViewDirectionX())
	demoVector[idx][second][ViewY] = float64(player.ViewDirectionY())
	// Health
	demoVector[idx][second][Health] = float64(player.Health())
	// Active Weapon
	if player.ActiveWeapon() != nil {
		demoVector[idx][second][ActiveWeapon] = float64(player.ActiveWeapon().Type)
	}
	// Blind
	demoVector[idx][second][Blind] = boolToFloat(player.IsBlinded())
	// Scoped
	demoVector[idx][second][Scoped] = boolToFloat(player.IsScoped())
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
	demoVector := [10][2700][23]float64{}

	// Find the start of the match
	for !p.GameState().IsMatchStarted() {
		p.ParseNextFrame()
	}

	playerMap := generatePlayerMap(p.GameState())

	// TODO: Register handler on each type of event

	var second int
	second = 0
	p.RegisterEventHandler(func(e events.WeaponFire) {
		fmt.Println(second)
	})

	// PARSE TO 45 Minutes
	startTime := p.CurrentTime()
out:
	for second = 0; second < 2700; second++ {
		for p.CurrentTime()-startTime < time.Second {
			EoF, err := p.ParseNextFrame()
			if err != nil {
				panic(err)
			}
			if !EoF {
				break out
			}
		}
		// One second has passed, get all data necessary
		players := p.GameState().Participants().Playing()
		for i := 0; i < len(players); i++ {
			p := players[i]
			idx := playerMap[p.Name]

			// Store pos and velocity
			updatePlayerState(p, second, idx, &demoVector)

		}

		startTime = p.CurrentTime()
	}

}
