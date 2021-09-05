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

// OUTPUT OF THIS PROGRAM IS A SERIES OF 2 NUMPY FILES PER DEMO
// 1. A log file, this contains the sequence data of what occured in the game. a vector of dimensions length x 10 x 23
//    the length is the number of seconds in the match, 10 is the number of players, with the first 5 being considered team 1
//    and the second 5 being considered team 2.
// 2. A round file, this contains the sequence of round outcomes. A vector of num. rounds x 2 where the first of the two is
//    the second that the round ended, and the second of the 2 is whether team 1 won or not. This includes an extra round at the
//    beginning for the knife round and a round at the end as padding which can be ignored.

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
	// TODO: translate and scale this relative to maps
	demoVector[second][idx][PosX] = player.Position().X
	demoVector[second][idx][PosY] = player.Position().Y
	demoVector[second][idx][PosZ] = player.Position().Z
	// Velocity
	demoVector[second][idx][VelocityX] = player.Velocity().X
	demoVector[second][idx][VelocityY] = player.Velocity().Y
	demoVector[second][idx][VelocityZ] = player.Velocity().Z
	// View
	demoVector[second][idx][ViewX] = float64(player.ViewDirectionX())
	demoVector[second][idx][ViewY] = float64(player.ViewDirectionY())
	// Health
	demoVector[second][idx][Health] = float64(player.Health())
	// Active Weapon
	if player.ActiveWeapon() != nil {
		demoVector[second][idx][ActiveWeapon] = float64(player.ActiveWeapon().Type)
	}
	// Blind
	demoVector[second][idx][Blind] = boolToFloat(player.IsBlinded())
	// Scoped
	demoVector[second][idx][Scoped] = boolToFloat(player.IsScoped())
	// Crouched
	demoVector[second][idx][Crouched] = boolToFloat(player.IsDucking() || player.IsDuckingInProgress())
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

func Flatten2D(arr [][]int16) []int16 {
	out := make([]int16, len(arr)*len(arr[0]))
	width := len(arr[0])
	for i := range arr {
		for j := range arr[i] {
			idx := i*width + j
			out[idx] = arr[i][j]
		}
	}
	return out
}

func validMap(mapName string) bool {
	validMaps := []string{"de_dust2", "de_inferno", "de_mirage", "de_overpass", "de_train", "de_vertigo"}
	for i := range validMaps {
		if validMaps[i] == mapName {
			return true
		}
	}
	return false
}

func ParseOneDemo(demoPath string, outputPath string, roundOutput string) {
	f, err := os.Open(demoPath)
	if err != nil {
		panic(err)
	}
	defer f.Close()
	p := dem.NewParser(f)
	defer p.Close()
	p.ParseHeader()

	if !validMap(p.Header().MapName) {
		fmt.Println(p.Header().MapName + " is not a valid map")
		return
	}

	roundVector := make([][]int16, 1)
	for i := range roundVector {
		roundVector[i] = make([]int16, 2)
	}
	var round int
	round = 0

	// Initialize to a 1 x 10 x 23 array
	// 1 second, 10 players, 23 state points
	demoVector := make([][][]float64, 1)
	for i := range demoVector {
		demoVector[i] = make([][]float64, 10)
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
					demoVector[second][idx][Incendiary] = boolToFloat(true)
				case common.EqMolotov:
					demoVector[second][idx][Incendiary] = boolToFloat(true)
				case common.EqSmoke:
					demoVector[second][idx][Smoke] = boolToFloat(true)
				case common.EqHE:
					demoVector[second][idx][Grenade] = boolToFloat(true)
				case common.EqFlash:
					demoVector[second][idx][Flash] = boolToFloat(true)
				case common.EqDecoy:
					demoVector[second][idx][Decoy] = boolToFloat(true)
				}
				// Guns
			} else {
				demoVector[second][idx][Shooting] = boolToFloat(true)

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
		demoVector[second][idx][Jumped] = boolToFloat(true)
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
		demoVector[second][idx][Reloaded] = boolToFloat(true)
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
		demoVector[second][idx][DefusedBomb] = boolToFloat(true)
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
		demoVector[second][idx][PlantedBomb] = boolToFloat(true)
	})

	// Store the second the round ended, and the winner of the round 1 if the first 5 players won, 0 otherwise
	p.RegisterEventHandler(func(e events.RoundEnd) {
		if e.WinnerState != nil {
			if len(e.WinnerState.Members()) > 0 {
				if idx, ok := playerMap[e.WinnerState.Members()[0].Name]; ok {
					roundVector[round][0] = int16(second)
					if idx < 5 {
						roundVector[round][1] = int16(1)
					} else {
						roundVector[round][1] = int16(0)
					}
					roundVector = append(roundVector, make([]int16, 2))
					round += 1
				} else {
					fmt.Println("No winner state")
					return
				}
			} else {
				fmt.Println("No winner state")
				return
			}
		} else {
			fmt.Println("No winner state")
			return
		}

	})

	// PARSE UP TO END
	startTime := p.CurrentTime()
	for moreFrames := true; moreFrames; moreFrames, err = p.ParseNextFrame() {
		if p.CurrentTime()-startTime >= time.Second {
			second += 1
			demoVector = append(demoVector, make([][]float64, 10))
			for j := range demoVector[second] {
				demoVector[second][j] = make([]float64, 23)
			}
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

	// Write the log data
	writer, _ := gonpy.NewFileWriter(outputPath)
	shape := []int{len(demoVector), 10, 23}
	writer.Shape = shape
	writer.Version = 2
	_ = writer.WriteFloat64(Flatten(demoVector))

	// Write the round data
	roundWriter, _ := gonpy.NewFileWriter(roundOutput)
	round_shape := []int{len(roundVector), 2}
	roundWriter.Shape = round_shape
	roundWriter.Version = 2

	_ = roundWriter.WriteInt16(Flatten2D(roundVector))
}

func main() {
	var demos []string
	demosPath := os.Args[1]
	sequencesPath := os.Args[2]
	roundsPath := os.Args[3]
	err := filepath.Walk(demosPath, func(path string, info os.FileInfo, err error) error {
		if filepath.Ext(path) == ".dem" {
			demos = append(demos, path)
		}
		return nil
	})
	if err != nil {
		panic(err)
	}
	for _, file := range demos[1:] {
		var extension = filepath.Ext(file)
		log_output := sequencesPath + "\\" + filepath.Base(file)[0:len(filepath.Base(file))-len(extension)] + ".npy"
		round_output := roundsPath + "\\" + filepath.Base(file)[0:len(filepath.Base(file))-len(extension)] + "_rounds.npy"
		fmt.Println(log_output)
		ParseOneDemo(file, log_output, round_output)
	}
}
