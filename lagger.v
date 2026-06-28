// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with this program. If not, see https://www.gnu.org/licenses/.

import net
import time
import math
import os
import flag
import rand as _
import json
import vnm

$if !windows {
	#include <signal.h>
}

fn C.signal(sig int, handler voidptr) voidptr

const state_colors_list = [
	'\x1b[38;5;34m',   // Green
	'\x1b[38;5;39m',   // Blue
	'\x1b[38;5;208m',  // Orange
	'\x1b[38;5;198m',  // Pink
	'\x1b[38;5;46m',   // Lime
	'\x1b[38;5;21m',   // Dark Blue
	'\x1b[38;5;165m',  // Purple
	'\x1b[38;5;226m',  // Yellow
	'\x1b[38;5;51m',   // Cyan
	'\x1b[38;5;93m',   // Violet
	'\x1b[38;5;202m',  // Red-Orange
	'\x1b[38;5;129m',  // Magenta
	'\x1b[38;5;118m',  // Light Green
	'\x1b[38;5;45m',   // Light Blue
	'\x1b[38;5;214m',  // Gold
	'\x1b[38;5;161m',  // Crimson
	'\x1b[38;5;82m',   // Bright Lime
	'\x1b[38;5;27m',   // Royal Blue
	'\x1b[38;5;201m',  // Fuchsia
	'\x1b[38;5;220m',  // Amber
	'\x1b[38;5;50m',   // Teal
	'\x1b[38;5;57m',   // Indigo
	'\x1b[38;5;196m',  // Pure Red
	'\x1b[38;5;135m',  // Orchid
	'\x1b[38;5;40m',   // Forest Green
	'\x1b[38;5;33m',   // Sky Blue
	'\x1b[38;5;200m',  // Hot Pink
	'\x1b[38;5;142m',  // Olive Yellow
	'\x1b[38;5;87m',   // Ice Blue
	'\x1b[38;5;99m',   // Slate Purple
	'\x1b[38;5;210m',  // Coral
	'\x1b[38;5;123m',  // Mint
]

struct TcpChunk {
	data         []u8
	arrival_time i64
}

struct UdpChunk {
	data         []u8
	dest         net.Addr
	arrival_time i64
}

struct ClientAddrHolder {
mut:
	addr_str string
}

struct ActiveCounter {
mut:
	count int
}

struct WaveConfig {
mut:
	min_lat        f64
	max_lat        f64
	sync           bool
	sync_inverse   bool
	pattern        string
	period         f64
	custom         []f64
	natural        bool
	jitter         f64
	correlation    f64
	inverse        bool
	last_lat       f64
	is_bad_state   bool
	loss_enabled   bool
	bandwidth_mbps f64
	analyze_only   bool
}

struct ClusteringModel {
mut:
	seq                  vnm.Sequential
	centroids            [][]vnm.Fnn
	centroid_initialized []bool
}

fn new_clustering_model() ClusteringModel {
	mut seq := vnm.new_sequential(.sgd)
	seq.set_normalize(false)
	
	seq.add(28, 64, .relu, 0.0, false)
	seq.add(64, 32, .relu, 0.0, false)
	seq.add(32, 32, .linear, 0.0, false)
	
	for i in 0 .. 64 {
		seq.net.layers[0].biases.data[i] = vnm.Fnn(0.01 * f64(i % 5 - 2))
		for j in 0 .. 28 {
			angle := f64(i * 28 + j) * 0.1
			seq.net.layers[0].weights.data[i * 28 + j] = vnm.Fnn(math.sin(angle) * 0.15)
		}
	}
	
	for i in 0 .. 32 {
		seq.net.layers[1].biases.data[i] = vnm.Fnn(0.01 * f64(i % 3 - 1))
		for j in 0 .. 64 {
			angle := f64(i * 64 + j) * 0.15
			seq.net.layers[1].weights.data[i * 64 + j] = vnm.Fnn(math.cos(angle) * 0.12)
		}
	}
	
	for i in 0 .. 32 {
		seq.net.layers[2].biases.data[i] = vnm.Fnn(0.0)
		for j in 0 .. 32 {
			angle := f64(i * 32 + j) * 0.2
			seq.net.layers[2].weights.data[i * 32 + j] = vnm.Fnn(math.sin(angle) * 0.2)
		}
	}

	mut centroids := [][]vnm.Fnn{len: 32}
	for i in 0 .. 32 {
		centroids[i] = []vnm.Fnn{len: 28}
		for j in 0 .. 28 {
			centroids[i][j] = vnm.Fnn(0.5)
		}
	}

	return ClusteringModel{
		seq:                  seq
		centroids:            centroids
		centroid_initialized: []bool{len: 32, init: false}
	}
}

fn (m ClusteringModel) clone() ClusteringModel {
	serialized := json.encode(m.seq.net)
	cloned_net := json.decode(vnm.NeuralNetwork, serialized) or { panic(err) }
	cloned_seq := vnm.Sequential{
		net: cloned_net
	}

	mut cloned_centroids := [][]vnm.Fnn{len: m.centroids.len}
	for i in 0 .. m.centroids.len { 
		cloned_centroids[i] = m.centroids[i].clone() 
	}

	return ClusteringModel{
		seq:                  cloned_seq
		centroids:            cloned_centroids
		centroid_initialized: m.centroid_initialized.clone()
	}
}

fn (mut m ClusteringModel) predict_and_update(input []vnm.Fnn, learning_rate f64) (int, f64) {
	input_tensor := vnm.new_tensor([28, 1], input)
	
	output_tensor := m.seq.predict(input_tensor) or { panic(err) }

	mut winner := 0
	mut max_val := -1e9
	for i in 0 .. 32 {
		val := f64(output_tensor.data[i])
		if val > max_val {
			max_val = val
			winner = i
		}
	}

	if learning_rate > 0.0 {
		mut target_data := []vnm.Fnn{len: 32, init: 0.0}
		target_data[winner] = 1.0
		target_tensor := vnm.new_tensor([32, 1], target_data)

		unsafe {
			trained_out := m.seq.net.train_step(input_tensor, target_tensor, vnm.Fnn(learning_rate)) or {
				panic(err)
			}
			trained_out.free()
		}
		
		if !m.centroid_initialized[winner] {
			for j in 0 .. 28 {
				m.centroids[winner][j] = input[j]
			}
			m.centroid_initialized[winner] = true
		} else {
			for j in 0 .. 28 {
				m.centroids[winner][j] += vnm.Fnn(learning_rate) * (input[j] - m.centroids[winner][j])
			}
		}
		target_tensor.free()
	}

	mut dist := 0.0
	for j in 0 .. 28 {
		diff := f64(input[j] - m.centroids[winner][j])
		dist += diff * diff
	}
	dist = math.sqrt(dist)

	mean_dist := dist / 5.2915
	confidence := math.exp(-mean_dist * 0.5) * 100.0

	output_tensor.free()
	input_tensor.free()

	return winner, confidence
}

struct StateToken {
mut:
	state string
	count int
}

fn (t StateToken) str() string {
	if t.count > 1 {
		return t.state + 'n' + t.count.str()
	}
	return t.state
}

struct SharedGrammar {
mut:
	pair_counts map[string]int
	merge_rules map[string]string
}

struct StateCompressor {
mut:
	history   []StateToken
	threshold int = 3
	grammar   shared SharedGrammar
}

fn (mut sc StateCompressor) squash() {
	if sc.history.len < 2 {
		return
	}
	mut i := 0
	for i < sc.history.len - 1 {
		if sc.history[i].state == sc.history[i + 1].state {
			sc.history[i].count += sc.history[i + 1].count
			sc.history.delete(i + 1)
		} else {
			i++
		}
	}
}

fn (mut sc StateCompressor) apply_merges() {
	if sc.history.len < 2 {
		return
	}
	mut i := 0
	for i < sc.history.len - 1 {
		p1 := sc.history[i].str()
		p2 := sc.history[i + 1].str()
		pair_key := p1 + '_' + p2

		mut has_rule := false
		mut merged := ''
		shared g := sc.grammar
		lock g {
			if pair_key in g.merge_rules {
				has_rule = true
				merged = g.merge_rules[pair_key]
			}
		}

		if has_rule {
			sc.history[i] = StateToken{
				state: merged
				count: 1
			}
			sc.history.delete(i + 1)
			if i > 0 {
				i--
			}
		} else {
			i++
		}
	}
	sc.squash()
}

fn (mut sc StateCompressor) add_state(new_state string) string {
	if sc.history.len > 0 && sc.history[sc.history.len - 1].state == new_state {
		sc.history[sc.history.len - 1].count++
	} else {
		sc.history << StateToken{
			state: new_state
			count: 1
		}
	}

	sc.apply_merges()

	if sc.history.len >= 2 {
		p1 := sc.history[sc.history.len - 2].str()
		p2 := sc.history[sc.history.len - 1].str()
		pair_key := p1 + '_' + p2

		mut trigger_merge := false
		mut merged_state := ''
		shared g := sc.grammar
		lock g {
			if g.pair_counts.len < 5000 && g.merge_rules.len < 5000 {
				g.pair_counts[pair_key]++
				if g.pair_counts[pair_key] >= sc.threshold && !(pair_key in g.merge_rules) {
					if p1.len + p2.len < 100 {
						merged_state = p1 + '+' + p2
						g.merge_rules[pair_key] = merged_state
						trigger_merge = true
					}
				}
			}
		}

		if trigger_merge {
			println('\x1b[33m[State Merger] New structural grammar rule: ' + p1 + ' + '
				+ p2 + ' -> ' + merged_state + '\x1b[0m')
			sc.apply_merges()
		}
	}

	if sc.history.len > 1000 {
		sc.history.delete(0)
	}

	if sc.history.len > 0 {
		return sc.history[sc.history.len - 1].str()
	}
	return new_state
}

struct ColorManager {
mut:
	state_colors map[string]string
}

struct LaggerConfig {
mut:
	model       ClusteringModel
	lag_configs map[string]WaveConfig
	merge_rules map[string]string
	pair_counts map[string]int
}

fn (mut cm ColorManager) get_color(state string) string {
	mut hash := 0
	for c in state {
		hash += int(c)
	}
	if hash < 0 {
		hash = -hash
	}
	idx := hash % state_colors_list.len
	return state_colors_list[idx]
}

struct TrafficAnalyzer {
mut:
	last_packet_time         i64
	last_print_time          i64
	packet_count             int
	bytes_accumulator        int
	sliding_window           []i64
	intervals                []i64
	entropies                []f64
	packet_sizes             []int
	rolling_entropy_vars     []f64
	byte_uniformities        []f64
	tail_patterns            []f64
	model                    ClusteringModel
	color_manager            shared ColorManager
	last_state               string
	last_winner              int
	last_confidence          f64
	conf_threshold           f64
	compressor               StateCompressor
	morph_mode               bool
}

fn is_ip_str(s string) bool {
	for c in s {
		if !((c >= 48 && c <= 57) || c == 46) {
			return false
		}
	}
	return true
}

fn deterministic_gaussian(step_ms i64) f64 {
	step := time.now().unix_milli() / step_ms
	mut x := u64(step)
	x = (x ^ 61) ^ (x >> 16)
	x *= 9
	x = x ^ (x >> 4)
	x *= 0x27d4eb2d
	x = x ^ (x >> 15)
	u1 := f64(x & 0xFFFFFFFF) / 4294967296.0
	mut y := u64(step + 0xFC3BD5C9)
	y = (y ^ 61) ^ (y >> 16)
	y *= 9
	y = y ^ (y >> 4)
	y *= 0x27d4eb2d
	y = y ^ (y >> 15)
	u2 := f64(y & 0xFFFFFFFF) / 4294967296.0
	safe_u1 := if u1 == 0.0 { 0.00001 } else { u1 }
	return math.sqrt(-2.0 * math.log(safe_u1)) * math.cos(2.0 * math.pi * u2)
}

struct FastRng {
mut:
	state u64
}

fn new_fast_rng() FastRng {
	return FastRng{
		state: u64(time.now().unix_nano())
	}
}

fn (mut r FastRng) next_f64() f64 {
	r.state += 0xa0761d6478bd642f
	mut temp := r.state ^ (r.state >> 30)
	temp *= 0xe7037ed1a0b428db
	mut temp2 := temp ^ (temp >> 27)
	temp2 *= 0x8ebc6af09c316535
	val := temp2 ^ (temp2 >> 31)
	return f64(val & 0xFFFFFFFFFFFF) / 281474976710656.0
}

fn (mut r FastRng) next_u8() u8 {
	r.state += 0xa0761d6478bd642f
	mut temp := r.state ^ (r.state >> 30)
	temp *= 0xe7037ed1a0b428db
	val := temp ^ (temp >> 27)
	return u8(val & 0xFF)
}

fn (mut r FastRng) int_in_range(min int, max int) int {
	if min >= max {
		return min
	}
	f := r.next_f64()
	return min + int(f * f64(max - min))
}

fn rand_gaussian(mut rng FastRng) f64 {
	mut u1 := rng.next_f64()
	for u1 == 0.0 {
		u1 = rng.next_f64()
	}
	u2 := rng.next_f64()
	return math.sqrt(-2.0 * math.log(u1)) * math.cos(2.0 * math.pi * u2)
}

fn rand_pareto(mut rng FastRng, minimum f64, alpha f64) f64 {
	u := rng.next_f64()
	safe_u := if u == 0.0 { 0.0001 } else { u }
	return minimum / math.pow(safe_u, 1.0 / alpha)
}

fn calculate_jitter(intervals []i64) f64 {
	if intervals.len < 2 {
		return 0.0
	}
	mut sum := 0.0
	for val in intervals {
		sum += f64(val)
	}
	mean := sum / f64(intervals.len)
	mut variance_sum := 0.0
	for val in intervals {
		diff := f64(val) - mean
		variance_sum += diff * diff
	}
	return math.sqrt(variance_sum / f64(intervals.len))
}

fn calculate_entropy(data []u8) f64 {
	if data.len == 0 {
		return 0.0
	}
	mut counts := [256]int{}
	for b in data {
		counts[b]++
	}
	mut sum := 0.0
	len_f := f64(data.len)
	for count in counts {
		if count > 0 {
			sum += f64(count) * math.log2(f64(count))
		}
	}
	entropy := math.log2(len_f) - (sum / len_f)
	return entropy / 8.0
}

fn calculate_renyi_entropy(data []u8) f64 {
	if data.len == 0 {
		return 0.0
	}
	mut counts := [256]int{}
	for b in data {
		counts[b]++
	}
	mut sum_sq := 0.0
	len_f := f64(data.len)
	for count in counts {
		p := f64(count) / len_f
		sum_sq += p * p
	}
	if sum_sq == 0.0 {
		return 0.0
	}
	entropy := -math.log2(sum_sq)
	return entropy / 8.0
}

fn calculate_min_entropy(data []u8) f64 {
	if data.len == 0 {
		return 0.0
	}
	mut counts := [256]int{}
	mut max_count := 0
	for b in data {
		counts[b]++
		if counts[b] > max_count {
			max_count = counts[b]
		}
	}
	p_max := f64(max_count) / f64(data.len)
	if p_max == 0.0 {
		return 0.0
	}
	entropy := -math.log2(p_max)
	return entropy / 8.0
}

fn calculate_rolling_entropy_variance(data []u8) f64 {
	if data.len < 128 {
		return 0.0
	}
	chunk_size := 64
	mut entropies := []f64{cap: 24}
	for i := 0; i + chunk_size <= data.len; i += chunk_size {
		entropies << calculate_entropy(data[i..i + chunk_size])
	}
	if entropies.len < 2 {
		return 0.0
	}
	mut sum := 0.0
	for ent in entropies {
		sum += ent
	}
	mean := sum / f64(entropies.len)
	mut var_sum := 0.0
	for ent in entropies {
		diff := ent - mean
		var_sum += diff * diff
	}
	return math.sqrt(var_sum / f64(entropies.len))
}

fn calculate_byte_uniformity(data []u8) f64 {
	if data.len < 16 {
		return 0.0
	}
	mut counts := [256]int{}
	for b in data {
		counts[b]++
	}
	mean := f64(data.len) / 256.0
	mut variance := 0.0
	for count in counts {
		diff := f64(count) - mean
		variance += diff * diff
	}
	norm_var := variance / f64(data.len)
	return math.max(0.0, 1.0 - (norm_var / 12.0))
}

fn calculate_tail_pattern(data []u8) f64 {
	if data.len < 16 {
		return 0.0
	}
	last_b := data[data.len - 1]
	mut matches := 0
	for i := 1; i <= 8; i++ {
		if data[data.len - i] == last_b {
			matches++
		}
	}
	return f64(matches) / 8.0
}

fn calculate_autocorr(sizes []int, lag int) f64 {
	if sizes.len <= lag {
		return 0.0
	}
	mut diff_sum := 0.0
	for i := 0; i < sizes.len - lag; i++ {
		mut diff := f64(sizes[i] - sizes[i + lag])
		if diff < 0.0 {
			diff = -diff
		}
		diff_sum += diff
	}
	avg_diff := diff_sum / f64(sizes.len - lag)
	return math.max(0.0, 1.0 - (avg_diff / 1500.0))
}

fn matches_filter(target_addr string, filter_parts []string) bool {
	if filter_parts.len == 0 {
		return true
	}
	for part in filter_parts {
		if part != '' && target_addr.contains(part) {
			return true
		}
	}
	return false
}

fn should_drop(mut cfg WaveConfig, mut rng FastRng) bool {
	if cfg.analyze_only {
		return false
	}
	if !cfg.loss_enabled {
		return false
	}
	r := rng.next_f64()
	if cfg.is_bad_state {
		if r < 0.15 {
			cfg.is_bad_state = false
		}
		return rng.next_f64() < 0.25
	} else {
		if r < 0.01 {
			cfg.is_bad_state = true
		}
		return rng.next_f64() < 0.002
	}
}

fn get_dynamic_latency(mut cfg WaveConfig, mut rng FastRng) f64 {
	base := (cfg.max_lat + cfg.min_lat) / 2.0
	amp := (cfg.max_lat - cfg.min_lat) / 2.0
	period_ms := cfg.period * 1000.0
	now_ms := time.now().unix_milli()
	mut val := 0.0
	match cfg.pattern {
		'sine' {
			phase := (f64(now_ms % i64(period_ms)) / period_ms) * 2.0 * math.pi
			val = base + amp * math.sin(phase)
		}
		'square' {
			half_period := period_ms / 2.0
			if now_ms % i64(period_ms) < i64(half_period) {
				val = cfg.min_lat
			} else {
				val = cfg.max_lat
			}
		}
		'triangle' {
			cycle := f64(now_ms % i64(period_ms)) / period_ms
			if cycle < 0.5 {
				val = cfg.min_lat + (cfg.max_lat - cfg.min_lat) * (cycle * 2.0)
			} else {
				val = cfg.max_lat - (cfg.max_lat - cfg.min_lat) * ((cycle - 0.5) * 2.0)
			}
		}
		'sawtooth' {
			cycle := f64(now_ms % i64(period_ms)) / period_ms
			val = cfg.min_lat + (cfg.max_lat - cfg.min_lat) * cycle
		}
		'random' {
			if cfg.sync || cfg.sync_inverse {
				step_ms := i64(period_ms / 10.0)
				noise_val := (deterministic_gaussian(if step_ms > 0 { step_ms } else { 100 }) + 3.0) / 6.0
				clamped := math.max(0.0, math.min(1.0, noise_val))
				val = cfg.min_lat + (cfg.max_lat - cfg.min_lat) * clamped
			} else {
				val = f64(rng.int_in_range(int(cfg.min_lat), int(cfg.max_lat)))
			}
		}
		'pulse' {
			cycle_ms := now_ms % i64(period_ms)
			if cycle_ms < 200 {
				val = cfg.max_lat
			} else {
				val = cfg.min_lat
			}
		}
		'custom' {
			if cfg.custom.len == 0 {
				val = cfg.min_lat
			} else if cfg.custom.len == 1 {
				val = cfg.custom[0]
			} else {
				cycle_progress := f64(now_ms % i64(period_ms)) / period_ms
				pos := cycle_progress * f64(cfg.custom.len)
				idx1 := int(math.floor(pos)) % cfg.custom.len
				idx2 := (idx1 + 1) % cfg.custom.len
				frac := pos - math.floor(pos)
				val = cfg.custom[idx1] * (1.0 - frac) + cfg.custom[idx2] * frac
			}
		}
		else {
			val = cfg.min_lat
		}
	}
	if cfg.inverse {
		val = cfg.max_lat + cfg.min_lat - val
	}
	if cfg.natural {
		mut random_variation := 0.0
		if cfg.sync || cfg.sync_inverse {
			raw_noise := deterministic_gaussian(50)
			multiplier := if cfg.inverse { -1.0 } else { 1.0 }
			random_variation = multiplier * raw_noise * cfg.jitter
		} else {
			random_variation = rand_gaussian(mut rng) * cfg.jitter
		}
		target_latency := val + random_variation
		if cfg.last_lat == 0.0 {
			cfg.last_lat = target_latency
		}
		current_latency := (cfg.correlation * cfg.last_lat) + ((1.0 - cfg.correlation) * target_latency)
		cfg.last_lat = current_latency
		return current_latency
	}
	return val
}

fn get_dynamic_latency_physical(mut cfg WaveConfig, mut rng FastRng, packet_len int) i64 {
	if cfg.analyze_only {
		return 0
	}
	base := get_dynamic_latency(mut cfg, mut rng)
	mut transmission_delay := 0.0
	if cfg.bandwidth_mbps > 0.0 {
		bandwidth_bps := cfg.bandwidth_mbps * 1_000_000.0
		packet_bits := f64(packet_len * 8)
		transmission_delay = (packet_bits / bandwidth_bps) * 1000.0
	}
	mut pareto_noise := 0.0
	if cfg.natural && rng.next_f64() < 0.15 {
		pareto_noise = rand_pareto(mut rng, 2.0, 1.4)
	}
	total := base + transmission_delay + pareto_noise
	if total < 5.0 {
		return 5
	}
	if total > 5000.0 {
		return 5000
	}
	return i64(math.round(total))
}

fn (mut ta TrafficAnalyzer) add_packet(data []u8, target_addr string) {
	now := time.now().unix_milli()
	size := data.len

	packet_entropy := calculate_entropy(data)
	ta.entropies << packet_entropy
	ta.packet_sizes << size

	p_roll_var := calculate_rolling_entropy_variance(data)
	ta.rolling_entropy_vars << p_roll_var

	p_uniform := calculate_byte_uniformity(data)
	ta.byte_uniformities << p_uniform

	p_tail := calculate_tail_pattern(data)
	ta.tail_patterns << p_tail

	if ta.sliding_window.len > 0 {
		interval := now - ta.sliding_window[ta.sliding_window.len - 1]
		ta.intervals << interval
	}
	ta.packet_count++
	ta.bytes_accumulator += size
	ta.sliding_window << now

	mut count_to_delete := 0
	for count_to_delete < ta.sliding_window.len {
		if now - ta.sliding_window[count_to_delete] <= 1000 {
			break
		}
		count_to_delete++
	}

	if count_to_delete > 0 {
		ta.sliding_window.delete_many(0, count_to_delete)
		if ta.intervals.len > ta.sliding_window.len {
			ta.intervals.delete_many(0, ta.intervals.len - ta.sliding_window.len)
		}
		if ta.entropies.len > ta.sliding_window.len {
			ta.entropies.delete_many(0, ta.entropies.len - ta.sliding_window.len)
		}
		if ta.packet_sizes.len > ta.sliding_window.len {
			ta.packet_sizes.delete_many(0, ta.packet_sizes.len - ta.sliding_window.len)
		}
		if ta.rolling_entropy_vars.len > ta.sliding_window.len {
			ta.rolling_entropy_vars.delete_many(0, ta.rolling_entropy_vars.len - ta.sliding_window.len)
		}
		if ta.byte_uniformities.len > ta.sliding_window.len {
			ta.byte_uniformities.delete_many(0, ta.byte_uniformities.len - ta.sliding_window.len)
		}
		if ta.tail_patterns.len > ta.sliding_window.len {
			ta.tail_patterns.delete_many(0, ta.tail_patterns.len - ta.sliding_window.len)
		}
	}

	if now - ta.last_print_time >= 500 {
		ta.analyze_state(target_addr)
		ta.last_print_time = now
	}

	if now - ta.last_packet_time > 1000 {
		ta.bytes_accumulator = size
		ta.packet_count = 1
		ta.intervals.clear()
		ta.entropies.clear()
		ta.packet_sizes.clear()
		ta.rolling_entropy_vars.clear()
		ta.byte_uniformities.clear()
		ta.tail_patterns.clear()
		ta.last_packet_time = now
	}
}

fn (mut ta TrafficAnalyzer) analyze_state(target_addr string) {
	pps := f64(ta.sliding_window.len)
	avg_size := if ta.packet_count > 0 {
		f64(ta.bytes_accumulator) / f64(ta.packet_count)
	} else {
		0.0
	}
	jitter := calculate_jitter(ta.intervals)

	mut sum_entropy := 0.0
	for ent in ta.entropies {
		sum_entropy += ent
	}
	avg_entropy := if ta.entropies.len > 0 { sum_entropy / f64(ta.entropies.len) } else { 0.0 }

	mut sum_size_diff_sq := 0.0
	for sz in ta.packet_sizes {
		diff := f64(sz) - avg_size
		sum_size_diff_sq += diff * diff
	}
	size_std_dev := if ta.packet_sizes.len > 0 {
		math.sqrt(sum_size_diff_sq / f64(ta.packet_sizes.len))
	} else {
		0.0
	}
	norm_size_std := math.min(1.0, size_std_dev / 500.0)

	mut large_packet_count := 0
	mut small_packet_count := 0
	mut med_packet_count := 0
	for sz in ta.packet_sizes {
		if sz > 1200 {
			large_packet_count++
		} else if sz < 128 {
			small_packet_count++
		} else {
			med_packet_count++
		}
	}
	large_ratio := if ta.packet_sizes.len > 0 { f64(large_packet_count) / f64(ta.packet_sizes.len) } else { 0.0 }
	small_packet_ratio := if ta.packet_sizes.len > 0 { f64(small_packet_count) / f64(ta.packet_sizes.len) } else { 0.0 }
	med_packet_ratio := if ta.packet_sizes.len > 0 { f64(med_packet_count) / f64(ta.packet_sizes.len) } else { 0.0 }

	mut sum_entropy_diff_sq := 0.0
	for ent in ta.entropies {
		diff := ent - avg_entropy
		sum_entropy_diff_sq += diff * diff
	}
	entropy_std_dev := if ta.entropies.len > 0 {
		math.sqrt(sum_entropy_diff_sq / f64(ta.entropies.len))
	} else {
		0.0
	}
	norm_entropy_std := math.min(1.0, entropy_std_dev / 0.5)

	mut short_interval_count := 0
	for val in ta.intervals {
		if val < 8 {
			short_interval_count++
		}
	}
	burst_ratio := if ta.intervals.len > 0 {
		f64(short_interval_count) / f64(ta.intervals.len)
	} else {
		0.0
	}

	mut align_8_count := 0
	mut align_16_count := 0
	mut align_32_count := 0
	mut align_64_count := 0
	for sz in ta.packet_sizes {
		if sz > 0 {
			if sz % 8 == 0 { align_8_count++ }
			if sz % 16 == 0 { align_16_count++ }
			if sz % 32 == 0 { align_32_count++ }
			if sz % 64 == 0 { align_64_count++ }
		}
	}
	align_8_ratio := if ta.packet_sizes.len > 0 { f64(align_8_count) / f64(ta.packet_sizes.len) } else { 0.0 }
	align_16_ratio := if ta.packet_sizes.len > 0 { f64(align_16_count) / f64(ta.packet_sizes.len) } else { 0.0 }
	align_32_ratio := if ta.packet_sizes.len > 0 { f64(align_32_count) / f64(ta.packet_sizes.len) } else { 0.0 }
	align_64_ratio := if ta.packet_sizes.len > 0 { f64(align_64_count) / f64(ta.packet_sizes.len) } else { 0.0 }

	mut sum_rev := 0.0
	for rev in ta.rolling_entropy_vars {
		sum_rev += rev
	}
	avg_rolling_entropy_var := if ta.rolling_entropy_vars.len > 0 {
		sum_rev / f64(ta.rolling_entropy_vars.len)
	} else {
		0.0
	}
	norm_rolling_entropy_var := math.min(1.0, avg_rolling_entropy_var / 0.4)

	mut sum_uni := 0.0
	for uni in ta.byte_uniformities {
		sum_uni += uni
	}
	avg_byte_uniformity := if ta.byte_uniformities.len > 0 {
		sum_uni / f64(ta.byte_uniformities.len)
	} else {
		0.0
	}
	
	mut sum_int := 0.0
	for val in ta.intervals {
		sum_int += f64(val)
	}
	avg_int := if ta.intervals.len > 0 { sum_int / f64(ta.intervals.len) } else { 0.0 }

	mut sum_int_diff_sq := 0.0
	for val in ta.intervals {
		diff := f64(val) - avg_int
		sum_int_diff_sq += diff * diff
	}
	int_std_dev := if ta.intervals.len > 0 {
		math.sqrt(sum_int_diff_sq / f64(ta.intervals.len))
	} else {
		0.0
	}

	mut skewness_iat := 0.0
	mut kurtosis_iat := 0.0
	if ta.intervals.len >= 3 && int_std_dev > 0.0 {
		mut skew_sum := 0.0
		mut kurt_sum := 0.0
		for val in ta.intervals {
			diff := f64(val) - avg_int
			skew_sum += diff * diff * diff
			kurt_sum += diff * diff * diff * diff
		}
		n := f64(ta.intervals.len)
		skewness_iat = (skew_sum / n) / (int_std_dev * int_std_dev * int_std_dev)
		kurtosis_iat = (kurt_sum / n) / (int_std_dev * int_std_dev * int_std_dev * int_std_dev) - 3.0
	}
	norm_skewness := math.min(1.0, math.max(0.0, (math.tanh(skewness_iat) + 1.0) / 2.0))
	norm_kurtosis := math.min(1.0, math.max(0.0, (math.tanh(kurtosis_iat) + 1.0) / 2.0))
	
	size_autocorr_1 := calculate_autocorr(ta.packet_sizes, 1)
	size_autocorr_2 := calculate_autocorr(ta.packet_sizes, 2)
	size_autocorr_3 := calculate_autocorr(ta.packet_sizes, 3)
	
	mut low_entropy_count := 0
	mut med_entropy_count := 0
	for ent in ta.entropies {
		if ent < 0.6 {
			low_entropy_count++
		} else if ent >= 0.6 && ent < 0.9 {
			med_entropy_count++
		}
	}
	low_entropy_ratio := if ta.entropies.len > 0 { f64(low_entropy_count) / f64(ta.entropies.len) } else { 0.0 }
	med_entropy_ratio := if ta.entropies.len > 0 { f64(med_entropy_count) / f64(ta.entropies.len) } else { 0.0 }
	
	mut constant_runs := 0
	for i := 0; i < ta.packet_sizes.len - 1; i++ {
		if ta.packet_sizes[i] == ta.packet_sizes[i + 1] {
			constant_runs++
		}
	}
	constant_run_ratio := if ta.packet_sizes.len > 1 {
		f64(constant_runs) / f64(ta.packet_sizes.len - 1)
	} else {
		0.0
	}
	
	mut sum_renyi := 0.0
	mut sum_min_ent := 0.0
	if ta.packet_sizes.len > 0 {
		mut sample_data := []u8{cap: 256}
		for sz in ta.packet_sizes {
			sample_data << u8(sz & 0xFF)
		}
		sum_renyi = calculate_renyi_entropy(sample_data)
		sum_min_ent = calculate_min_entropy(sample_data)
	}
	
	mut acc_sum := 0.0
	if ta.intervals.len > 1 {
		for i := 0; i < ta.intervals.len - 1; i++ {
			mut acc := f64(ta.intervals[i + 1] - ta.intervals[i])
			if acc < 0 {
				acc = -acc
			}
			acc_sum += acc
		}
		acc_sum /= f64(ta.intervals.len - 1)
	}
	norm_pps_derivative := math.min(1.0, acc_sum / 1000.0)
	mut sum_tail_patterns := 0.0
	for p_tail in ta.tail_patterns {
		sum_tail_patterns += p_tail
	}
	avg_tail_pattern := if ta.tail_patterns.len > 0 {
		sum_tail_patterns / f64(ta.tail_patterns.len)
	} else {
		0.0
	}

	norm_pps := math.min(1.0, pps / 120.0)
	norm_size := math.min(1.0, avg_size / 1500.0)
	norm_jitter := math.min(1.0, jitter / 500.0)
	
	input := [
		vnm.Fnn(norm_pps),                 // [0]
		vnm.Fnn(norm_size),                // [1]
		vnm.Fnn(norm_jitter),              // [2]
		vnm.Fnn(avg_entropy),              // [3]
		vnm.Fnn(norm_size_std),            // [4]
		vnm.Fnn(large_ratio),              // [5]
		vnm.Fnn(norm_entropy_std),          // [6]
		vnm.Fnn(burst_ratio),              // [7]
		vnm.Fnn(align_16_ratio),           // [8]
		vnm.Fnn(norm_rolling_entropy_var), // [9]
		vnm.Fnn(avg_byte_uniformity),      // [10]
		vnm.Fnn(norm_skewness),            // [11]
		vnm.Fnn(norm_kurtosis),            // [12]
		vnm.Fnn(size_autocorr_1),          // [13]
		vnm.Fnn(low_entropy_ratio),        // [14]
		vnm.Fnn(med_entropy_ratio),        // [15]
		vnm.Fnn(align_8_ratio),            // [16]
		vnm.Fnn(align_32_ratio),           // [17]
		vnm.Fnn(size_autocorr_2),          // [18]
		vnm.Fnn(size_autocorr_3),          // [19]
		vnm.Fnn(small_packet_ratio),       // [20]
		vnm.Fnn(med_packet_ratio),         // [21]
		vnm.Fnn(constant_run_ratio),       // [22]
		vnm.Fnn(sum_renyi),                // [23]
		vnm.Fnn(norm_pps_derivative),      // [24]
		vnm.Fnn(avg_tail_pattern),         // [25]
		vnm.Fnn(align_64_ratio),           // [26]
		vnm.Fnn(sum_min_ent),              // [27]
	]

	learning_rate := if ta.morph_mode {
		0.0
	} else if ta.last_state == '' {
		0.1
	} else {
		0.02
	}
	
	winner, confidence := ta.model.predict_and_update(input, learning_rate)

	ta.last_confidence = confidence
	ta.last_winner = winner

	if confidence >= ta.conf_threshold {
		compressed_state := ta.compressor.add_state(winner.str())

		if compressed_state != ta.last_state {
			ta.last_state = compressed_state

			shared cm := ta.color_manager
			mut color := ''
			lock cm {
				color = cm.get_color(compressed_state)
			}

			println('${color}[Deep NN Transition] ${target_addr} -> Mode ${compressed_state} (Confidence: ${int(ta.last_confidence)}%) | Metrics -> PPS: ${int(pps)} Size: ${int(avg_size)}B Jitter: ${int(jitter)}ms Entropy: ${avg_entropy:.2f} Autocorr1: ${size_autocorr_1:.2f}\x1b[0m')
		}
	}
}

fn dial_socks5(socks5_addr string, target_addr string) !&net.TcpConn {
	mut conn := net.dial_tcp(socks5_addr)!
	conn.set_read_timeout(15 * time.second)
	conn.set_write_timeout(15 * time.second)
	conn.write([u8(0x05), 0x01, 0x00])!
	mut buf := []u8{len: 2}
	conn.read(mut buf)!
	if buf[0] != 0x05 || buf[1] != 0x00 {
		conn.close() or {}
		return error('SOCKS5 handshake failed')
	}
	parts := target_addr.split(':')
	if parts.len < 2 {
		conn.close() or {}
		return error('invalid target address')
	}
	port := parts[parts.len - 1].int()
	host := parts[..parts.len - 1].join(':')
	mut req := []u8{}
	req << 0x05
	req << 0x01
	req << 0x00
	is_ip := is_ip_str(host)
	if is_ip {
		req << 0x01
		ip_parts := host.split('.')
		if ip_parts.len == 4 {
			for p_part in ip_parts {
				req << u8(p_part.int())
			}
		} else {
			conn.close() or {}
			return error('invalid ipv4')
		}
	} else {
		req << 0x03
		req << u8(host.len)
		for c in host {
			req << u8(c)
		}
	}
	req << u8((port >> 8) & 0xff)
	req << u8(port & 0xff)
	conn.write(req)!
	mut resp := []u8{len: 10}
	conn.read(mut resp)!
	if resp[0] != 0x05 || resp[1] != 0x00 {
		conn.close() or {}
		return error('SOCKS5 connection failed')
	}
	return conn
}

fn dial_target(target string, upstream string) !&net.TcpConn {
	if upstream != '' {
		return dial_socks5(upstream, target)
	}
	return net.dial_tcp(target)
}

fn send_udp_with_delay(mut conn &net.UdpConn, packet UdpChunk, shared active_threads ActiveCounter) {
	now := time.now().unix_milli()
	remaining := packet.arrival_time - now
	if remaining > 0 {
		sleep_dur := if remaining > 5000 { i64(5000) } else { remaining }
		time.sleep(time.Duration(sleep_dur) * time.millisecond)
	}
	conn.write_to(packet.dest, packet.data) or {}
	lock active_threads {
		active_threads.count--
	}
}

fn read_tcp(mut src &net.TcpConn, ch chan TcpChunk, cfg WaveConfig, target_addr string, direction string, mut analyzer &TrafficAnalyzer, lag_configs map[string]WaveConfig, lag_states []string, conf_threshold f64, target_filter []string, morph_mode bool) {
	mut local_cfg := cfg
	mut buf := []u8{len: 4096}
	mut rng := new_fast_rng()
	mut next_send_time := time.now().unix_milli()
	mut packet_index := 0
	for {
		bytes_read := src.read(mut buf) or { break }
		if bytes_read == 0 {
			break
		}
		packet_index++
		is_handshake := packet_index < 15

		is_matched := matches_filter(target_addr, target_filter)

		if is_matched && direction == 'downstream' {
			analyzer.add_packet(buf[..bytes_read], target_addr)
		}

		mut active_cfg := local_cfg

		if !is_matched {
			active_cfg.min_lat = 0
			active_cfg.max_lat = 0
			active_cfg.loss_enabled = false
		} else {
			mut should_apply_lag := true
			if analyzer.last_confidence < conf_threshold {
				should_apply_lag = false
			}

			if should_apply_lag {
				if lag_configs.len > 0 {
					state_str := analyzer.last_state
					if state_str in lag_configs {
						active_cfg = lag_configs[state_str]
						if local_cfg.analyze_only {
							active_cfg.analyze_only = true
						}
					} else {
						active_cfg.min_lat = 0
						active_cfg.max_lat = 0
						active_cfg.loss_enabled = false
					}
				} else if lag_states.len > 0 {
					mut matches := false
					for ls in lag_states {
						if analyzer.last_state == ls || analyzer.last_state.contains(ls) {
							matches = true
							break
						}
					}
					if !matches {
						active_cfg.min_lat = 0
						active_cfg.max_lat = 0
						active_cfg.loss_enabled = false
					}
				}
			} else {
				active_cfg.min_lat = 0
				active_cfg.max_lat = 0
				active_cfg.loss_enabled = false
			}
		}

		if should_drop(mut active_cfg, mut rng) {
			continue
		}

		mut packets_to_send := [][]u8{}
		should_morph := morph_mode && !is_handshake && analyzer.last_confidence >= conf_threshold

		if should_morph {
			target_size := int(f64(analyzer.model.centroids[analyzer.last_winner][1]) * 1500.0)
			mut safe_target_size := target_size
			if safe_target_size < 512 {
				safe_target_size = 512
			}
			if safe_target_size > 1460 {
				safe_target_size = 1460
			}

			data := buf[..bytes_read].clone()
			if data.len < safe_target_size {
				mut padded := data.clone()
				for padded.len < safe_target_size {
					padded << rng.next_u8()
				}
				packets_to_send << padded
			} else if data.len > safe_target_size {
				mut offset := 0
				mut frag_count := 0
				for offset < data.len {
					frag_count++
					mut chunk_size := safe_target_size
					if frag_count >= 4 || offset + chunk_size > data.len {
						chunk_size = data.len - offset
					}
					mut chunk := data[offset..offset + chunk_size].clone()
					if chunk.len < safe_target_size && frag_count < 4 {
						for chunk.len < safe_target_size {
							chunk << rng.next_u8()
						}
					}
					packets_to_send << chunk
					offset += chunk_size
					if frag_count >= 4 {
						break
					}
				}
			} else {
				packets_to_send << data
			}
		} else {
			packets_to_send << buf[..bytes_read].clone()
		}

		for pkt_data in packets_to_send {
			delay := get_dynamic_latency_physical(mut active_cfg, mut rng, pkt_data.len)
			mut arrival := time.now().unix_milli() + delay

			if should_morph {
				target_pps := f64(analyzer.model.centroids[analyzer.last_winner][0]) * 120.0
				mut interval := i64(0)
				if target_pps > 1.0 {
					interval = i64(1000.0 / target_pps)
				}
				now := time.now().unix_milli()
				if next_send_time < now {
					next_send_time = now
				}
				if next_send_time > now + 500 {
					next_send_time = now + 500
				}
				arrival = next_send_time + delay
				next_send_time += interval
			}

			mut pushed := false
			select {
				ch <- TcpChunk{
					data:         pkt_data
					arrival_time: arrival
				} {
					pushed = true
				}
				500 * time.millisecond {}
			}
			if !pushed {
				break
			}
		}
	}
	ch.close()
	src.close() or {}
}

fn write_tcp_delayed(mut dst &net.TcpConn, ch chan TcpChunk) {
	for {
		chunk := <-ch or { break }
		for {
			now := time.now().unix_milli()
			remaining := chunk.arrival_time - now
			if remaining <= 0 {
				break
			}
			sleep_dur := if remaining > 5000 { i64(5000) } else { remaining }
			time.sleep(time.Duration(sleep_dur) * time.millisecond)
		}
		dst.write(chunk.data) or { break }
	}
	dst.close() or {}
}

fn handle_tcp_connection(mut client net.TcpConn, target string, up_cfg WaveConfig, down_cfg WaveConfig, upstream string, shared cm ColorManager, model ClusteringModel, lag_configs map[string]WaveConfig, lag_states []string, conf_threshold f64, target_filter []string, shared grammar SharedGrammar, morph_mode bool) {
	mut server_conn := dial_target(target, upstream) or {
		eprintln('[TCP] Failed to connect to server: ${err}')
		client.close() or {}
		return
	}
	server_conn.set_read_timeout(120 * time.second)
	server_conn.set_write_timeout(120 * time.second)

	ch_to_server := chan TcpChunk{cap: 1024}
	ch_to_client := chan TcpChunk{cap: 1024}
	mut server_ref := server_conn

	mut analyzer_up := TrafficAnalyzer{
		model:           model.clone()
		color_manager:   cm
		last_state:      ''
		last_winner:     0
		last_confidence: 0.0
		conf_threshold:  conf_threshold
		compressor:      StateCompressor{
			threshold: 3
			grammar:   grammar
		}
		morph_mode:      morph_mode
	}
	mut analyzer_down := TrafficAnalyzer{
		model:           model.clone()
		color_manager:   cm
		last_state:      ''
		last_winner:     0
		last_confidence: 0.0
		conf_threshold:  conf_threshold
		compressor:      StateCompressor{
			threshold: 3
			grammar:   grammar
		}
		morph_mode:      morph_mode
	}

	mut threads := []thread{}
	threads << spawn read_tcp(mut &client, ch_to_server, up_cfg, target, 'upstream', mut
		&analyzer_up, lag_configs, lag_states, conf_threshold, target_filter, morph_mode)
	threads << spawn read_tcp(mut server_ref, ch_to_client, down_cfg, target, 'downstream', mut
		&analyzer_down, lag_configs, lag_states, conf_threshold, target_filter, morph_mode)
	threads << spawn write_tcp_delayed(mut server_ref, ch_to_server)
	threads << spawn write_tcp_delayed(mut &client, ch_to_client)
	threads.wait()
}

fn start_tcp_proxy(port int, target string, up_cfg WaveConfig, down_cfg WaveConfig, upstream string, shared cm ColorManager, model ClusteringModel, lag_configs map[string]WaveConfig, lag_states []string, conf_threshold f64, target_filter []string, shared grammar SharedGrammar, morph_mode bool) {
	_ = port
	mut listener := net.listen_tcp(.ip, '127.0.0.1:${port}') or {
		eprintln('Failed to start TCP proxy: ${err}')
		return
	}
	println('Lagger TCP proxy listening on port ${port} forwarding to ${target}')
	for {
		mut client_conn := listener.accept() or { continue }
		client_conn.set_read_timeout(120 * time.second)
		client_conn.set_write_timeout(120 * time.second)
		spawn handle_tcp_connection(mut client_conn, target, up_cfg, down_cfg, upstream,
			shared cm, model, lag_configs, lag_states, conf_threshold, target_filter,
			shared grammar, morph_mode)
	}
}

fn forward_udp_server_to_client(mut target_conn &net.UdpConn, mut proxy_conn &net.UdpConn, shared holder ClientAddrHolder, cfg WaveConfig, shared cm ColorManager, model ClusteringModel, lag_configs map[string]WaveConfig, lag_states []string, conf_threshold f64, target_filter []string, shared grammar SharedGrammar, morph_mode bool, shared active_threads ActiveCounter) {
	mut local_cfg := cfg
	mut buf := []u8{len: 2048}
	mut analyzer := TrafficAnalyzer{
		model:           model.clone()
		color_manager:   cm
		last_state:      ''
		last_winner:     0
		last_confidence: 0.0
		conf_threshold:  conf_threshold
		compressor:      StateCompressor{
			threshold: 3
			grammar:   grammar
		}
		morph_mode:      morph_mode
	}
	mut rng := new_fast_rng()
	mut last_dest_str := ''
	mut cached_dest := net.Addr{}
	mut next_send_time := time.now().unix_milli()
	mut packet_index := 0
	for {
		bytes_read, _ := target_conn.read(mut buf) or { continue }
		if bytes_read == 0 {
			continue
		}
		packet_index++
		is_handshake := packet_index < 15

		is_matched := matches_filter('UDP_Downstream_Server', target_filter)
		if is_matched {
			analyzer.add_packet(buf[..bytes_read], 'UDP_Downstream_Server')
		}

		mut active_cfg := local_cfg
		state_str := analyzer.last_state

		if !is_matched {
			active_cfg.min_lat = 0
			active_cfg.max_lat = 0
			active_cfg.loss_enabled = false
		} else {
			mut should_apply_lag := true
			if analyzer.last_confidence < conf_threshold {
				should_apply_lag = false
			}

			if should_apply_lag {
				if lag_configs.len > 0 {
					if state_str in lag_configs {
						active_cfg = lag_configs[state_str]
						if local_cfg.analyze_only {
							active_cfg.analyze_only = true
						}
					} else {
						active_cfg.min_lat = 0
						active_cfg.max_lat = 0
						active_cfg.loss_enabled = false
					}
				} else if lag_states.len > 0 {
					mut matches := false
					for ls in lag_states {
						if analyzer.last_state == ls || analyzer.last_state.contains(ls) {
							matches = true
							break
						}
					}
					if !matches {
						active_cfg.min_lat = 0
						active_cfg.max_lat = 0
						active_cfg.loss_enabled = false
					}
				}
			} else {
				active_cfg.min_lat = 0
				active_cfg.max_lat = 0
				active_cfg.loss_enabled = false
			}
		}

		if should_drop(mut active_cfg, mut rng) {
			continue
		}

		mut packets_to_send := [][]u8{}
		should_morph := morph_mode && !is_handshake && analyzer.last_confidence >= conf_threshold

		if should_morph {
			target_size := int(f64(analyzer.model.centroids[analyzer.last_winner][1]) * 1500.0)
			mut safe_target_size := target_size
			if safe_target_size < 256 {
				safe_target_size = 256
			}
			if safe_target_size > 1400 {
				safe_target_size = 1400
			}
			data := buf[..bytes_read].clone()
			if data.len < safe_target_size {
				mut padded := data.clone()
				for padded.len < safe_target_size {
					padded << rng.next_u8()
				}
				packets_to_send << padded
			} else {
				packets_to_send << data
			}
		} else {
			packets_to_send << buf[..bytes_read].clone()
		}

		for pkt_data in packets_to_send {
			delay := get_dynamic_latency_physical(mut active_cfg, mut rng, pkt_data.len)
			mut arrival := time.now().unix_milli() + delay
			if rng.next_f64() < 0.02 {
				arrival -= 8
			}

			if should_morph {
				target_pps := f64(analyzer.model.centroids[analyzer.last_winner][0]) * 120.0
				mut interval := i64(0)
				if target_pps > 1.0 {
					interval = i64(1000.0 / target_pps)
				}
				now := time.now().unix_milli()
				if next_send_time < now {
					next_send_time = now
				}
				if next_send_time > now + 500 {
					next_send_time = now + 500
				}
				arrival = next_send_time + delay
				next_send_time += interval
			}

			mut dest_str := ''
			lock holder {
				dest_str = holder.addr_str
			}
			if dest_str != '' {
				mut dest := net.Addr{}
				if dest_str == last_dest_str {
					dest = cached_dest
				} else {
					dest_addrs := net.resolve_addrs_fuzzy(dest_str, .udp) or { continue }
					if dest_addrs.len == 0 {
						continue
					}
					dest = dest_addrs[0]
					last_dest_str = dest_str
					cached_dest = dest
				}

				now := time.now().unix_milli()
				if arrival <= now + 3 {
					proxy_conn.write_to(dest, pkt_data) or {}
				} else {
					mut can_spawn := false
					lock active_threads {
						if active_threads.count < 100 {
							active_threads.count++
							can_spawn = true
						}
					}
					if can_spawn {
						spawn send_udp_with_delay(mut proxy_conn, UdpChunk{
							data:         pkt_data
							dest:         dest
							arrival_time: arrival
						}, shared active_threads)
					} else {
						proxy_conn.write_to(dest, pkt_data) or {}
					}
				}
			}
		}
	}
}

fn start_udp_proxy(port int, target string, up_cfg WaveConfig, down_cfg WaveConfig, shared cm ColorManager, model ClusteringModel, lag_configs map[string]WaveConfig, lag_states []string, conf_threshold f64, target_filter []string, shared grammar SharedGrammar, morph_mode bool) {
	_ = port
	mut proxy_conn := net.listen_udp('127.0.0.1:${port}') or {
		eprintln('Failed to start UDP proxy: ${err}')
		return
	}
	defer {
		proxy_conn.close() or {}
	}
	println('Lagger UDP proxy listening on port ${port} forwarding to ${target}')
	mut target_conn := net.dial_udp(target) or {
		eprintln('Failed to dial UDP target: ${err}')
		return
	}
	defer {
		target_conn.close() or {}
	}

	dest_addrs := net.resolve_addrs_fuzzy(target, .udp) or {
		eprintln('Failed to resolve UDP target: ${err}')
		return
	}
	if dest_addrs.len == 0 {
		eprintln('No addresses found for target ${target}')
		return
	}
	dest := dest_addrs[0]

	shared active_threads := &ActiveCounter{
		count: 0
	}

	shared holder := &ClientAddrHolder{}
	spawn forward_udp_server_to_client(mut target_conn, mut proxy_conn, shared holder,
		down_cfg, shared cm, model, lag_configs, lag_states, conf_threshold, target_filter,
		shared grammar, morph_mode, shared active_threads)
	mut local_up_cfg := up_cfg
	mut buf := []u8{len: 2048}
	mut rng := new_fast_rng()

	mut analyzer := TrafficAnalyzer{
		model:           model.clone()
		color_manager:   cm
		last_state:      ''
		last_winner:     0
		last_confidence: 0.0
		conf_threshold:  conf_threshold
		compressor:      StateCompressor{
			threshold: 3
			grammar:   grammar
		}
		morph_mode:      morph_mode
	}
	mut next_send_time := time.now().unix_milli()
	mut packet_index := 0

	for {
		bytes_read, client_addr := proxy_conn.read(mut buf) or { continue }
		if bytes_read == 0 {
			continue
		}
		packet_index++
		is_handshake := packet_index < 15

		is_matched := matches_filter('UDP_Upstream_Client', target_filter)
		if is_matched {
			analyzer.add_packet(buf[..bytes_read], 'UDP_Upstream_Client')
		}

		mut active_up_cfg := local_up_cfg
		state_str := analyzer.last_state

		if !is_matched {
			active_up_cfg.min_lat = 0
			active_up_cfg.max_lat = 0
			active_up_cfg.loss_enabled = false
		} else {
			mut should_apply_lag := true
			if analyzer.last_confidence < conf_threshold {
				should_apply_lag = false
			}

			if should_apply_lag {
				if lag_configs.len > 0 {
					if state_str in lag_configs {
						active_up_cfg = lag_configs[state_str]
						if local_up_cfg.analyze_only {
							active_up_cfg.analyze_only = true
						}
					} else {
						active_up_cfg.min_lat = 0
						active_up_cfg.max_lat = 0
						active_up_cfg.loss_enabled = false
					}
				} else if lag_states.len > 0 {
					mut matches := false
					for ls in lag_states {
						if analyzer.last_state == ls || analyzer.last_state.contains(ls) {
							matches = true
							break
						}
					}
					if !matches {
						active_up_cfg.min_lat = 0
						active_up_cfg.max_lat = 0
						active_up_cfg.loss_enabled = false
					}
				}
			} else {
				active_up_cfg.min_lat = 0
				active_up_cfg.max_lat = 0
				active_up_cfg.loss_enabled = false
			}
		}

		if should_drop(mut active_up_cfg, mut rng) {
			continue
		}

		client_str := client_addr.str()
		lock holder {
			holder.addr_str = client_str
		}

		mut packets_to_send := [][]u8{}
		should_morph := morph_mode && !is_handshake && analyzer.last_confidence >= conf_threshold

		if should_morph {
			target_size := int(f64(analyzer.model.centroids[analyzer.last_winner][1]) * 1500.0)
			mut safe_target_size := target_size
			if safe_target_size < 256 {
				safe_target_size = 256
			}
			if safe_target_size > 1400 {
				safe_target_size = 1400
			}
			data := buf[..bytes_read].clone()
			if data.len < safe_target_size {
				mut padded := data.clone()
				for padded.len < safe_target_size {
					padded << rng.next_u8()
				}
				packets_to_send << padded
			} else {
				packets_to_send << data
			}
		} else {
			packets_to_send << buf[..bytes_read].clone()
		}

		for pkt_data in packets_to_send {
			delay := get_dynamic_latency_physical(mut active_up_cfg, mut rng, pkt_data.len)
			mut arrival := time.now().unix_milli() + delay
			if rng.next_f64() < 0.02 {
				arrival -= 8
			}

			if should_morph {
				target_pps := f64(analyzer.model.centroids[analyzer.last_winner][0]) * 120.0
				mut interval := i64(0)
				if target_pps > 1.0 {
					interval = i64(1000.0 / target_pps)
				}
				now := time.now().unix_milli()
				if next_send_time < now {
					next_send_time = now
				}
				if next_send_time > now + 500 {
					next_send_time = now + 500
				}
				arrival = next_send_time + delay
				next_send_time += interval
			}

			now := time.now().unix_milli()
			if arrival <= now + 3 {
				target_conn.write_to(dest, pkt_data) or {}
			} else {
				mut can_spawn := false
				lock active_threads {
					if active_threads.count < 100 {
						active_threads.count++
						can_spawn = true
					}
				}
				if can_spawn {
					spawn send_udp_with_delay(mut target_conn, UdpChunk{
						data:         pkt_data
						dest:         dest
						arrival_time: arrival
					}, shared active_threads)
				} else {
					target_conn.write_to(dest, pkt_data) or {}
				}
			}
		}
	}
}

fn start_socks5_proxy(port int, up_cfg WaveConfig, down_cfg WaveConfig, upstream string, shared cm ColorManager, model ClusteringModel, lag_configs map[string]WaveConfig, lag_states []string, conf_threshold f64, target_filter []string, shared grammar SharedGrammar, morph_mode bool) {
	_ = port
	mut listener := net.listen_tcp(.ip, '127.0.0.1:${port}') or {
		eprintln('Failed to start SOCKS5 proxy: ${err}')
		return
	}
	println('Lagger SOCKS5 proxy listening on port ${port}')
	for {
		mut client_conn := listener.accept() or { continue }
		client_conn.set_read_timeout(120 * time.second)
		client_conn.set_write_timeout(120 * time.second)
		spawn handle_socks5(mut client_conn, up_cfg, down_cfg, upstream, shared cm, model,
			lag_configs, lag_states, conf_threshold, target_filter, shared grammar, morph_mode)
	}
}

fn handle_socks5(mut client net.TcpConn, up_cfg WaveConfig, down_cfg WaveConfig, upstream string, shared cm ColorManager, model ClusteringModel, lag_configs map[string]WaveConfig, lag_states []string, conf_threshold f64, target_filter []string, shared grammar SharedGrammar, morph_mode bool) {
	mut buf := []u8{len: 512}
	bytes_read := client.read(mut buf) or { return }
	if bytes_read < 2 || buf[0] != 0x05 {
		client.close() or {}
		return
	}
	client.write([u8(0x05), 0x00]) or { return }
	req_bytes := client.read(mut buf) or { return }
	if req_bytes < 7 || buf[0] != 0x05 || buf[1] != 0x01 {
		client.close() or {}
		return
	}
	atyp := buf[3]
	mut target_host := ''
	mut target_port := 0
	match atyp {
		0x01 {
			if req_bytes < 10 {
				client.close() or {}
				return
			}
			target_host = '${buf[4]}.${buf[5]}.${buf[6]}.${buf[7]}'
			target_port = int((u32(buf[8]) << 8) | buf[9])
		}
		0x03 {
			len_host := int(buf[4])
			if req_bytes < 5 + len_host + 2 {
				client.close() or {}
				return
			}
			target_host = buf[5..5 + len_host].bytestr()
			target_port = int((u32(buf[5 + len_host]) << 8) | buf[5 + len_host + 1])
		}
		else {
			client.write([u8(0x05), 0x08, 0x00, 0x01, 0, 0, 0, 0, 0, 0]) or {}
			client.close() or {}
			return
		}
	}
	target_addr := '${target_host}:${target_port}'
	mut server_conn := dial_target(target_addr, upstream) or {
		client.write([u8(0x05), 0x05, 0x00, 0x01, 0, 0, 0, 0, 0, 0]) or {}
		client.close() or {}
		return
	}
	server_conn.set_read_timeout(120 * time.second)
	server_conn.set_write_timeout(120 * time.second)

	client.write([u8(0x05), 0x00, 0x00, 0x01, 0, 0, 0, 0, 0, 0]) or {
		server_conn.close() or {}
		client.close() or {}
		return
	}
	ch_to_server := chan TcpChunk{cap: 1024}
	ch_to_client := chan TcpChunk{cap: 1024}
	mut server_ref := server_conn

	mut analyzer_up := TrafficAnalyzer{
		model:           model.clone()
		color_manager:   cm
		last_state:      ''
		last_winner:     0
		last_confidence: 0.0
		conf_threshold:  conf_threshold
		compressor:      StateCompressor{
			threshold: 3
			grammar:   grammar
		}
		morph_mode:      morph_mode
	}
	mut analyzer_down := TrafficAnalyzer{
		model:           model.clone()
		color_manager:   cm
		last_state:      ''
		last_winner:     0
		last_confidence: 0.0
		conf_threshold:  conf_threshold
		compressor:      StateCompressor{
			threshold: 3
			grammar:   grammar
		}
		morph_mode:      morph_mode
	}

	mut threads := []thread{}
	threads << spawn read_tcp(mut &client, ch_to_server, up_cfg, target_addr, 'upstream', mut
		&analyzer_up, lag_configs, lag_states, conf_threshold, target_filter, morph_mode)
	threads << spawn read_tcp(mut server_ref, ch_to_client, down_cfg, target_addr, 'downstream', mut
		&analyzer_down, lag_configs, lag_states, conf_threshold, target_filter, morph_mode)
	threads << spawn write_tcp_delayed(mut server_ref, ch_to_server)
	threads << spawn write_tcp_delayed(mut &client, ch_to_client)
	threads.wait()
}

fn parse_custom(s string) []f64 {
	if s == '' {
		return []f64{}
	}
	parts := s.split(',')
	mut res := []f64{}
	for p_val in parts {
		val := p_val.trim_space().f64()
		res << val
	}
	return res
}

fn main() {
	$if !windows {
		unsafe {
			C.signal(C.SIGPIPE, C.SIG_IGN)
		}
	}

	mut fp := flag.new_flag_parser(os.args)
	fp.application('lagger')
	fp.version('1.5.0')
	fp.description('Dynamic Latency Simulator & Side-Channel Behavior Analyzer Proxy')
	fp.skip_executable()
	up_pattern := fp.string('up-pattern', `p`, 'sine', '')
	up_min := fp.float('up-min', `n`, 50.0, '')
	up_max := fp.float('up-max', `x`, 150.0, '')
	up_period := fp.float('up-period', `t`, 5.0, '')
	up_custom_str := fp.string('up-custom', `c`, '', '')
	up_natural := fp.bool('up-natural', `u`, false, '')
	up_jitter := fp.float('up-jitter', `j`, 15.0, '')
	up_correlation := fp.float('up-correlation', `k`, 0.75, '')
	up_bandwidth := fp.float('up-bandwidth', `z`, 50.0, 'Upstream bandwidth in Mbps')
	down_pattern := fp.string('down-pattern', `o`, 'sine', '')
	down_min := fp.float('down-min', `i`, 50.0, '')
	down_max := fp.float('down-max', `a`, 150.0, '')
	down_period := fp.float('down-period', `e`, 5.0, '')
	down_custom_str := fp.string('down-custom', `y`, '', '')
	down_natural := fp.bool('down-natural', `d`, false, '')
	down_jitter := fp.float('down-jitter', `f`, 15.0, '')
	down_correlation := fp.float('down-correlation', `g`, 0.75, '')
	down_bandwidth := fp.float('down-bandwidth', `q`, 50.0, 'Downstream bandwidth in Mbps')

	sync_mode := fp.bool('sync', `s`, false, '')
	sync_inverse := fp.bool('sync-inverse', `b`, false, '')
	port := fp.int('port', `l`, 1080, '')
	target := fp.string('target', `r`, '127.0.0.1:8080', '')
	proto := fp.string('proto', `m`, 'socks5', '')
	upstream := fp.string('upstream', `w`, '', '')

	analyze_mode := fp.bool('analyze', `a`, false, 'Run in analyze-only mode (no lag/loss, prints states)')
	load_path := fp.string('load-model', `f`, '', 'Path to a .lgr JSON file to load neural network weights & lag configs')
	lag_on_str := fp.string('lag-on', `g`, '', 'Only apply lag/loss on these comma-separated states (e.g. "2,3")')

	save_path := fp.string('save-model', `v`, 'model.lgr', 'Filename to save the neural network weights & lag configs when exiting')
	conf_threshold := fp.float('conf-threshold', `c`, 0.0, 'Minimum neural network confidence percentage to trigger lag and display transitions (0 to 100)')

	target_filter_raw := fp.string('target-filter', `t`, '', 'Only analyze/lag targets matching this comma-separated filter (e.g. "telegram,149.154")')
	morph_mode := fp.bool('morph', `X`, false, 'Morph traffic sizes and timing to match the loaded model')

	_ := fp.finalize() or {
		eprintln(err)
		println(fp.usage())
		return
	}

	mut target_filter := []string{}
	if target_filter_raw != '' {
		for part in target_filter_raw.split(',') {
			trimmed := part.trim_space()
			if trimmed != '' {
				target_filter << trimmed
			}
		}
	}

	mut model := new_clustering_model()
	mut lag_configs := map[string]WaveConfig{}

	shared grammar := &SharedGrammar{
		pair_counts: map[string]int{}
		merge_rules: map[string]string{}
	}

	if load_path != '' {
		println('Loading neural network weights and lag workspaces from ${load_path}...')
		content := os.read_file(load_path) or {
			eprintln('Failed to read model file: ${err}')
			return
		}
		config := json.decode(LaggerConfig, content) or {
			eprintln('Failed to decode Lagger config JSON: ${err}')
			return
		}
		model = config.model
		lag_configs = config.lag_configs.clone()
		lock grammar {
			grammar.merge_rules = config.merge_rules.clone()
			grammar.pair_counts = config.pair_counts.clone()
		}
		println('Model, ${lag_configs.len} state-specific lag workspaces, and ${config.merge_rules.len} learned grammar rules loaded successfully!')
	}

	os.signal_opt(.int, fn [save_path, model, lag_configs, grammar] (_ os.Signal) {
		println('\n[SIGINT] Interrupted by user. Saving configuration to ${save_path}...')

		mut final_configs := lag_configs.clone()
		if final_configs.len == 0 {
			final_configs['3'] = WaveConfig{
				min_lat:      100.0
				max_lat:      300.0
				pattern:      'sine'
				period:       5.0
				natural:      true
				jitter:       15.0
				loss_enabled: true
			}
		}

		mut saved_rules := map[string]string{}
		mut saved_counts := map[string]int{}
		lock grammar {
			saved_rules = grammar.merge_rules.clone()
			saved_counts = grammar.pair_counts.clone()
		}

		config := LaggerConfig{
			model:       model
			lag_configs: final_configs
			merge_rules: saved_rules
			pair_counts: saved_counts
		}

		model_json := json.encode_pretty(config)
		os.write_file(save_path, model_json) or {
			eprintln('Error saving model file: ${err}')
			exit(1)
		}
		println('[SYSTEM] Saved model, customizable lag templates, and ${saved_rules.len} learned grammar rules to ${save_path} successfully. Exiting.')
		exit(0)
	}) or {
		eprintln('Failed to register SIGINT handler: ${err}')
	}

	mut lag_states := []string{}
	if lag_on_str != '' {
		parts := lag_on_str.split(',')
		for p in parts {
			lag_states << p.trim_space()
		}
		println('Targeted Lagging enabled! Only lagging on states: ${lag_states}')
	}

	if conf_threshold > 0.0 {
		println('Confidence Filter enabled! Only lagging and printing analysis when Neural Network is at least ${conf_threshold}% confident.')
	}

	if target_filter_raw != '' {
		println('Target Filter enabled! Only analyzing and lagging domains/IPs matching: "${target_filter_raw}"')
	}

	mut up_cfg := WaveConfig{
		pattern:        up_pattern
		min_lat:        up_min
		max_lat:        up_max
		period:         up_period
		custom:         parse_custom(up_custom_str)
		natural:        up_natural
		jitter:         up_jitter
		correlation:    up_correlation
		sync:           sync_mode
		sync_inverse:   sync_inverse
		inverse:        false
		is_bad_state:   false
		loss_enabled:   up_natural
		bandwidth_mbps: up_bandwidth
		analyze_only:   analyze_mode
	}

	mut down_cfg := WaveConfig{
		pattern:        down_pattern
		min_lat:        down_min
		max_lat:        down_max
		period:         down_period
		custom:         parse_custom(down_custom_str)
		natural:        down_natural
		jitter:         down_jitter
		correlation:    down_correlation
		sync:           sync_mode
		sync_inverse:   sync_inverse
		inverse:        false
		is_bad_state:   false
		loss_enabled:   down_natural
		bandwidth_mbps: down_bandwidth
		analyze_only:   analyze_mode
	}

	if sync_mode {
		down_cfg.pattern = up_cfg.pattern
		down_cfg.period = up_cfg.period
		down_cfg.custom = up_cfg.custom
		down_cfg.natural = up_cfg.natural
		down_cfg.jitter = up_cfg.jitter
		down_cfg.correlation = up_cfg.correlation
		down_cfg.loss_enabled = up_cfg.loss_enabled
		down_cfg.bandwidth_mbps = up_cfg.bandwidth_mbps
	} else if sync_inverse {
		down_cfg.pattern = up_cfg.pattern
		down_cfg.period = up_cfg.period
		down_cfg.custom = up_cfg.custom
		down_cfg.natural = up_cfg.natural
		down_cfg.jitter = up_cfg.jitter
		down_cfg.correlation = up_cfg.correlation
		down_cfg.loss_enabled = up_cfg.loss_enabled
		down_cfg.bandwidth_mbps = up_cfg.bandwidth_mbps
		down_cfg.inverse = true
	}

	shared color_manager := &ColorManager{
		state_colors: map[string]string{}
	}

	if analyze_mode {
		println('Running in ANALYZE-ONLY mode. Lagging is bypassed.')
	}

	if morph_mode {
		if load_path == '' {
			println('\x1b[31m[WARNING] Morph mode is enabled but no model (--load-model) is loaded. Using default centroids.\x1b[0m')
		} else {
			println('\x1b[32m[SYSTEM] Morphing Mode ACTIVE. Reshaping packet sizes with high-entropy random padding and timing pacing to look like the loaded model.\x1b[0m')
		}
	}

	println('Starting Lagger Dynamic Latency & Behavior Analyzer')
	if proto == 'tcp' {
		start_tcp_proxy(port, target, up_cfg, down_cfg, upstream, shared color_manager,
			model, lag_configs, lag_states, conf_threshold, target_filter, shared grammar,
			morph_mode)
	} else if proto == 'udp' {
		start_udp_proxy(port, target, up_cfg, down_cfg, shared color_manager, model,
			lag_configs, lag_states, conf_threshold, target_filter, shared grammar, morph_mode)
	} else if proto == 'socks5' {
		start_socks5_proxy(port, up_cfg, down_cfg, upstream, shared color_manager, model,
			lag_configs, lag_states, conf_threshold, target_filter, shared grammar, morph_mode)
	} else {
		eprintln('Unknown protocol: ${proto}')
	}
}
