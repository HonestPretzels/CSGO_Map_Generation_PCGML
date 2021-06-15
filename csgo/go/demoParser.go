package main

import (
	"fmt"
	"os"
	"path/filepath"
	"time"

	"github.com/kshedden/gonpy"
	dem "github.com/markus-wa/demoinfocs-golang/v2/pkg/demoinfocs"
	"github.com/markus-wa/demoinfocs-golang/v2/pkg/demoinfocs/common"
	"github.com/markus-wa/demoinfocs-golang/v2/pkg/demoinfocs/events"
)

//TODO:
// - Use arguments for input and output path of demo
// - Parse maps somehow
// - Combine with web scraping

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
	DefusedBomb
)

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

func updatePlayerState(player *common.Player, second int, idx int, demoVector [][][]float64) {
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
	// Crouched
	demoVector[idx][second][Crouched] = boolToFloat(player.IsDucking() || player.IsDuckingInProgress())
}

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

func ParseOneDemo(demoPath string, outputPath string) {
	f, err := os.Open(demoPath)
	if err != nil {
		panic(err)
	}
	defer f.Close()
	p := dem.NewParser(f)
	defer p.Close()

	// Initialize to a 10 x 1 x 23 array
	// 10 players, 1 second, 23 state points
	demoVector := make([][][]float64, 10)
	for i := range demoVector {
		demoVector[i] = make([][]float64, 1)
		for j := range demoVector[i] {
			demoVector[i][j] = make([]float64, 23)
		}
	}

	// Find the start of the match and initialize place markers
	for !p.GameState().IsMatchStarted() {
		p.ParseNextFrame()
	}
	var second int
	second = 0
	playerMap := generatePlayerMap(p.GameState())

	// Handle shots fired and util thrown
	p.RegisterEventHandler(func(e events.WeaponFire) {
		// Get player idx
		player := e.Shooter
		var idx int
		if player != nil {
			idx = playerMap[player.Name]
		} else {
			return
		}

		// Check weapon type
		if e.Weapon.Class() != common.EqClassEquipment {
			// Utility
			if e.Weapon.Class() == common.EqClassGrenade {

				switch e.Weapon.Type {
				case common.EqIncendiary:
					demoVector[idx][second][Incendiary] = boolToFloat(true)
				case common.EqMolotov:
					demoVector[idx][second][Incendiary] = boolToFloat(true)
				case common.EqSmoke:
					demoVector[idx][second][Smoke] = boolToFloat(true)
				case common.EqHE:
					demoVector[idx][second][Grenade] = boolToFloat(true)
				case common.EqFlash:
					demoVector[idx][second][Flash] = boolToFloat(true)
				case common.EqDecoy:
					demoVector[idx][second][Decoy] = boolToFloat(true)
				}
				// Guns
			} else {
				demoVector[idx][second][Shooting] = boolToFloat(true)

			}
		}
	})

	// Handle jumps
	p.RegisterEventHandler(func(e events.PlayerJump) {
		// Get player idx
		player := e.Player
		var idx int
		if player != nil {
			idx = playerMap[player.Name]
		} else {
			return
		}
		demoVector[idx][second][Jumped] = boolToFloat(true)
	})

	// Handle Reloads
	p.RegisterEventHandler(func(e events.WeaponReload) {
		// Get player idx
		player := e.Player
		var idx int
		if player != nil {
			idx = playerMap[player.Name]
		} else {
			return
		}
		demoVector[idx][second][Reloaded] = boolToFloat(true)
	})

	// Handle Bomb Defuses
	p.RegisterEventHandler(func(e events.BombDefused) {
		// Get player idx
		player := e.Player
		var idx int
		if player != nil {
			idx = playerMap[player.Name]
		} else {
			return
		}
		demoVector[idx][second][DefusedBomb] = boolToFloat(true)
	})

	// Handle Bomb Plants
	p.RegisterEventHandler(func(e events.BombPlanted) {
		// Get player idx
		player := e.Player
		var idx int
		if player != nil {
			idx = playerMap[player.Name]
		} else {
			return
		}
		demoVector[idx][second][PlantedBomb] = boolToFloat(true)
	})

	// PARSE UP TO 45 Minutes
	startTime := p.CurrentTime()
	for moreFrames := true; moreFrames; moreFrames, err = p.ParseNextFrame() {
		if p.CurrentTime()-startTime >= time.Second {
			for i := range demoVector {
				demoVector[i] = append(demoVector[i], make([]float64, 23))
			}
			second += 1
			startTime = p.CurrentTime()

			players := p.GameState().Participants().Playing()
			for i := 0; i < len(players); i++ {
				p := players[i]
				idx := playerMap[p.Name]

				// Store pos and velocity
				updatePlayerState(p, second, idx, demoVector)

			}
		}
		if err != nil {
			panic(err)
		}
	}

	writer, _ := gonpy.NewFileWriter(outputPath)
	shape := []int{10, 2700, 23}
	writer.Shape = shape
	writer.Version = 2
	_ = writer.WriteFloat64(Flatten(demoVector))
}

func main() {
	var demos []string
	root := "E:/Projects/GRAIL_PCGML_tmaurer_summer_2021/csgo"
	err := filepath.Walk(root+"/logs/demos/", func(path string, info os.FileInfo, err error) error {
		demos = append(demos, path)
		return nil
	})
	if err != nil {
		panic(err)
	}
	for _, file := range demos[1:] {
		var extension = filepath.Ext(file)
		output := root + "/vectors/" + filepath.Base(file)[0:len(filepath.Base(file))-len(extension)] + ".npy"
		fmt.Println(output)
		ParseOneDemo(file, output)
	}
}
