import net
import time
import math
import os
import flag
import rand

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

struct WaveConfig {
	min_lat      f64
	max_lat      f64
	sync         bool
	sync_inverse bool
mut:
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

fn rand_gaussian() f64 {
	mut u1 := rand.f64()
	for u1 == 0.0 {
		u1 = rand.f64()
	}
	u2 := rand.f64()
	return math.sqrt(-2.0 * math.log(u1)) * math.cos(2.0 * math.pi * u2)
}

fn rand_pareto(minimum f64, alpha f64) f64 {
	u := rand.f64()
	safe_u := if u == 0.0 { 0.0001 } else { u }
	return minimum / math.pow(safe_u, 1.0 / alpha)
}

fn should_drop(mut cfg WaveConfig) bool {
	if !cfg.loss_enabled {
		return false
	}
	r := rand.f64()
	if cfg.is_bad_state {
		if r < 0.15 {
			cfg.is_bad_state = false
		}
		return rand.f64() < 0.25
	} else {
		if r < 0.01 {
			cfg.is_bad_state = true
		}
		return rand.f64() < 0.002
	}
}

fn get_dynamic_latency(mut cfg WaveConfig) f64 {
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
				val = f64(rand.int_in_range(int(cfg.min_lat), int(cfg.max_lat)) or { int(cfg.min_lat) })
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
			random_variation = rand_gaussian() * cfg.jitter
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

fn get_dynamic_latency_physical(mut cfg WaveConfig, packet_len int) i64 {
	base := get_dynamic_latency(mut cfg)
	
	bandwidth_bps := cfg.bandwidth_mbps * 1_000_000.0
	packet_bits := f64(packet_len * 8)
	transmission_delay := (packet_bits / bandwidth_bps) * 1000.0
	
	mut pareto_noise := 0.0
	if cfg.natural && rand.f64() < 0.15 {
		pareto_noise = rand_pareto(2.0, 1.4)
	}
	total := base + transmission_delay + pareto_noise
	if total < 5.0 {
		return 5
	}
	return i64(math.round(total))
}

fn dial_socks5(socks5_addr string, target_addr string) !&net.TcpConn {
	mut conn := net.dial_tcp(socks5_addr)!
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

fn start_tcp_proxy(port int, target string, up_cfg WaveConfig, down_cfg WaveConfig, upstream string) {
	_ = port
	mut listener := net.listen_tcp(.ip, '127.0.0.1:${port}') or {
		eprintln('Failed to start TCP proxy: ${err}')
		return
	}
	println('Lagger TCP proxy listening on port ${port} forwarding to ${target}')
	for {
		mut client_conn := listener.accept() or { continue }
		println('[TCP] Connection accepted.')
		spawn handle_tcp_connection(mut client_conn, target, up_cfg, down_cfg, upstream)
	}
}

fn handle_tcp_connection(mut client net.TcpConn, target string, up_cfg WaveConfig, down_cfg WaveConfig, upstream string) {
	mut server_conn := dial_target(target, upstream) or {
		eprintln('[TCP] Failed to connect to server: ${err}')
		client.close() or {}
		return
	}
	ch_to_server := chan TcpChunk{}
	ch_to_client := chan TcpChunk{}
	mut server_ref := server_conn

	mut threads := []thread{}
	threads << spawn read_tcp(mut &client, ch_to_server, up_cfg)
	threads << spawn read_tcp(mut server_ref, ch_to_client, down_cfg)
	threads << spawn write_tcp_delayed(mut server_ref, ch_to_server)
	threads << spawn write_tcp_delayed(mut &client, ch_to_client)
	threads.wait()
}

fn read_tcp(mut src &net.TcpConn, ch chan TcpChunk, cfg WaveConfig) {
	mut local_cfg := cfg
	mut buf := []u8{len: 4096}
	for {
		bytes_read := src.read(mut buf) or { break }
		if bytes_read == 0 {
			break
		}
		if should_drop(mut local_cfg) {
			continue
		}
		data := buf[..bytes_read].clone()
		delay := get_dynamic_latency_physical(mut local_cfg, bytes_read)
		arrival := time.now().unix_milli() + delay
		
		ch <- TcpChunk{ 
			data:         data
			arrival_time: arrival
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
			if remaining > 2 {
				time.sleep(time.Duration(remaining - 1) * time.millisecond)
			} else {
				for time.now().unix_milli() < chunk.arrival_time {}
				break
			}
		}
		dst.write(chunk.data) or { break }
	}
	dst.close() or {}
}

fn start_udp_proxy(port int, target string, up_cfg WaveConfig, down_cfg WaveConfig) {
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
	shared holder := &ClientAddrHolder{}
	spawn forward_udp_server_to_client(mut target_conn, mut proxy_conn, shared holder,
		down_cfg)
	mut local_up_cfg := up_cfg
	mut buf := []u8{len: 2048}
	for {
		bytes_read, client_addr := proxy_conn.read(mut buf) or { continue }
		if bytes_read == 0 {
			continue
		}
		if should_drop(mut local_up_cfg) {
			continue
		}
		client_str := client_addr.str()
		lock holder {
			holder.addr_str = client_str
		}
		packet_data := buf[..bytes_read].clone()
		delay := get_dynamic_latency_physical(mut local_up_cfg, bytes_read)
		mut arrival := time.now().unix_milli() + delay
		if rand.f64() < 0.02 {
			arrival -= 8
		}
		dest_addrs := net.resolve_addrs_fuzzy(target, .udp) or { continue }
		if dest_addrs.len == 0 {
			continue
		}
		dest := dest_addrs[0]
		spawn send_udp_with_delay(mut target_conn, UdpChunk{
			data:         packet_data
			dest:         dest
			arrival_time: arrival
		})
	}
}

fn forward_udp_server_to_client(mut target_conn &net.UdpConn, mut proxy_conn &net.UdpConn, shared holder ClientAddrHolder, cfg WaveConfig) {
	mut local_cfg := cfg
	mut buf := []u8{len: 2048}
	for {
		bytes_read, _ := target_conn.read(mut buf) or { continue }
		if bytes_read == 0 {
			continue
		}
		if should_drop(mut local_cfg) {
			continue
		}
		packet_data := buf[..bytes_read].clone()
		delay := get_dynamic_latency_physical(mut local_cfg, bytes_read)
		mut arrival := time.now().unix_milli() + delay
		if rand.f64() < 0.02 {
			arrival -= 8
		}
		mut dest_str := ''
		lock holder {
			dest_str = holder.addr_str
		}
		if dest_str != '' {
			dest_addrs := net.resolve_addrs_fuzzy(dest_str, .udp) or { continue }
			if dest_addrs.len == 0 {
				continue
			}
			dest := dest_addrs[0]
			spawn send_udp_with_delay(mut proxy_conn, UdpChunk{
				data:         packet_data
				dest:         dest
				arrival_time: arrival
			})
		}
	}
}

fn send_udp_with_delay(mut conn &net.UdpConn, packet UdpChunk) {
	for {
		now := time.now().unix_milli()
		remaining := packet.arrival_time - now
		if remaining <= 0 {
			break
		}
		if remaining > 2 {
			time.sleep(time.Duration(remaining - 1) * time.millisecond)
		} else {
			for time.now().unix_milli() < packet.arrival_time {}
			break
		}
	}
	conn.write_to(packet.dest, packet.data) or { return }
}

fn handle_socks5(mut client net.TcpConn, up_cfg WaveConfig, down_cfg WaveConfig, upstream string) {
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
	client.write([u8(0x05), 0x00, 0x00, 0x01, 0, 0, 0, 0, 0, 0]) or {
		server_conn.close() or {}
		client.close() or {}
		return
	}
	ch_to_server := chan TcpChunk{}
	ch_to_client := chan TcpChunk{}
	mut server_ref := server_conn

	mut threads := []thread{}
	threads << spawn read_tcp(mut &client, ch_to_server, up_cfg)
	threads << spawn read_tcp(mut server_ref, ch_to_client, down_cfg)
	threads << spawn write_tcp_delayed(mut server_ref, ch_to_server)
	threads << spawn write_tcp_delayed(mut &client, ch_to_client)
	threads.wait()
}

fn start_socks5_proxy(port int, up_cfg WaveConfig, down_cfg WaveConfig, upstream string) {
	_ = port
	mut listener := net.listen_tcp(.ip, '127.0.0.1:${port}') or {
		eprintln('Failed to start SOCKS5 proxy: ${err}')
		return
	}
	println('Lagger SOCKS5 proxy listening on port ${port}')
	for {
		mut client_conn := listener.accept() or { continue }
		spawn handle_socks5(mut client_conn, up_cfg, down_cfg, upstream)
	}
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
	mut fp := flag.new_flag_parser(os.args)
	fp.application('lagger')
	fp.version('1.3.0')
	fp.description('Dynamic Latency Simulator Proxy')
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
	_ := fp.finalize() or {
		eprintln(err)
		println(fp.usage())
		return
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

	println('Starting Lagger Latency Proxy')
	if proto == 'tcp' {
		start_tcp_proxy(port, target, up_cfg, down_cfg, upstream)
	} else if proto == 'udp' {
		start_udp_proxy(port, target, up_cfg, down_cfg)
	} else if proto == 'socks5' {
		start_socks5_proxy(port, up_cfg, down_cfg, upstream)
	} else {
		eprintln('Unknown protocol: ${proto}')
	}
}