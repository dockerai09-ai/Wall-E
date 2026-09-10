#!/usr/bin/env bash
#
# startup.sh — start every Wall-E service with one command.
#
#   ./startup.sh              dev (default): API server + dashboard with hot reload
#   ./startup.sh prod         build once, then serve API + dashboard from one port
#   ./startup.sh podman       podman compose up -d --build (docker: same flow with Docker)
#   ./startup.sh stop         stop what a previous --detach or docker run started
#   ./startup.sh status       show what is running and which ports are taken
#
# Works with macOS bash 3.2 as well as bash 4+/Linux.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"
STATE_DIR="$ROOT/.startup"
UI_PORT=5173

# ---------------------------------------------------------------- output ----
c_bold=$'\033[1m'; c_red=$'\033[31m'; c_grn=$'\033[32m'; c_ylw=$'\033[33m'; c_off=$'\033[0m'
[ -t 1 ] || { c_bold=''; c_red=''; c_grn=''; c_ylw=''; c_off=''; }
info() { printf '%s>%s %s\n' "$c_grn" "$c_off" "$*"; }
warn() { printf '%s!%s %s\n' "$c_ylw" "$c_off" "$*" >&2; }
die()  { printf '%sx%s %s\n' "$c_red" "$c_off" "$*" >&2; exit 1; }

usage() {
  cat <<'USAGE'
Usage: ./startup.sh [dev|prod|docker|stop|status] [options]

Modes
  dev       API server (tsx watch) + Vite dashboard with hot reload   [default]
            Dashboard http://localhost:5173, API http://localhost:3001
  prod      npm run build, then node server/dist/index.js
            Dashboard + API on http://localhost:3001
  podman    podman compose up -d --build (starts/creates the podman machine if needed)
            Dashboard + API on http://localhost:3001
  docker    the same compose flow with Docker, for machines without podman
  stop      stop services started with --detach and/or compose (podman or docker)
  status    show running services and port usage

Options
  --detach       run in the background; logs in .startup/<name>.log
  --lan          expose to your LAN (dev: Vite --host; podman/docker: HOST_BIND=0.0.0.0;
                 prod: HOST=0.0.0.0). Only on a trusted network.
  --open         open the dashboard in a browser once it answers
  --no-install   skip dependency install (npm ci) and .env creation
  --with-nativ   also build and launch the Nativ local-inference app from
                 external/nativ (macOS 26+, Apple silicon, Xcode + xcodegen)
  -h, --help     this text

Environment
  PORT               API port (default 3001; read from .env when present)
  NODE_BIN           directory holding a Node 20.18–24 node/npm to prefer
  CONTAINER_ENGINE   podman | docker (default: podman when installed, else docker)
USAGE
}

# --------------------------------------------------------------- options ----
MODE=dev; DETACH=0; LAN=0; OPEN=0; INSTALL=1; WITH_NATIV=0
while [ $# -gt 0 ]; do
  case "$1" in
    dev|prod|podman|docker|stop|status) MODE=$1 ;;
    --detach)      DETACH=1 ;;
    --lan)         LAN=1 ;;
    --open)        OPEN=1 ;;
    --no-install)  INSTALL=0 ;;
    --with-nativ)  WITH_NATIV=1 ;;
    -h|--help)     usage; exit 0 ;;
    *) die "Unknown argument: $1 (try --help)" ;;
  esac
  shift
done

env_port=''
[ -f .env ] && env_port="$(sed -n 's/^PORT=//p' .env | tail -1 | tr -d '"'"'"' \r')"
API_PORT="${PORT:-${env_port:-3001}}"

# ------------------------------------------------------------------ node ----
# Prints the version when $1 is a runnable Node within the supported range
# (package.json engines: >=20.18.0 <25). A binary that cannot start (for
# instance Homebrew's node after a simdutf upgrade) fails `-v` and is skipped.
node_ok() {
  local v major minor rest
  v="$("$1" -v 2>/dev/null)" || return 1
  v=${v#v}; major=${v%%.*}; rest=${v#*.}; minor=${rest%%.*}
  [ "$major" -lt 25 ] || return 1
  if [ "$major" -gt 20 ] || { [ "$major" -eq 20 ] && [ "$minor" -ge 18 ]; }; then
    printf '%s' "$v"
    return 0
  fi
  return 1
}

pick_node() {
  # Honour nvm + .nvmrc when present, then try known locations in order.
  if [ -s "${NVM_DIR:-$HOME/.nvm}/nvm.sh" ]; then
    # shellcheck disable=SC1091
    . "${NVM_DIR:-$HOME/.nvm}/nvm.sh" >/dev/null 2>&1 || true
    nvm use --silent >/dev/null 2>&1 || true
  fi
  local current cand v
  current="$(dirname "$(command -v node 2>/dev/null || echo /nonexistent/node)")"
  for cand in "${NODE_BIN:-}" "$current" /opt/homebrew/opt/node@22/bin /opt/homebrew/opt/node@24/bin \
              /opt/homebrew/opt/node@20/bin /usr/local/bin /usr/bin; do
    [ -n "$cand" ] && [ -x "$cand/node" ] && [ -x "$cand/npm" ] || continue
    if v="$(node_ok "$cand/node")"; then
      export PATH="$cand:$PATH"
      info "Node v$v ($cand)"
      return 0
    fi
  done
  die "No working Node.js 20.18–24 with npm found. Install one (brew install node@22) or set NODE_BIN=/path/to/bin."
}

# ----------------------------------------------------------------- ports ----
port_busy() { command -v lsof >/dev/null 2>&1 && lsof -nP -iTCP:"$1" -sTCP:LISTEN >/dev/null 2>&1; }
port_owner() { lsof -nP -iTCP:"$1" -sTCP:LISTEN 2>/dev/null | awk 'NR>1 {print $1 " (pid " $2 ")"}' | sort -u | paste -sd', ' -; }
require_free() {
  local p
  for p in "$@"; do
    # Written as `if`, not `port_busy && die`: under set -e a failing && list
    # as the function's last command would abort the script when the port is free.
    if port_busy "$p"; then
      die "Port $p is already in use by $(port_owner "$p"). Run ./startup.sh stop, or set PORT."
    fi
  done
}

# Probes use `localhost`, not 127.0.0.1: Vite listens on ::1 only, and curl
# tries both address families for localhost, exactly as a browser does.
wait_for_url() { # url, timeout-seconds
  local i=0
  until curl -fsS -o /dev/null "$1" 2>/dev/null; do
    i=$((i + 1)); [ "$i" -ge "${2:-90}" ] && return 1
    sleep 1
  done
}

open_url() {
  if command -v open >/dev/null 2>&1; then open "$1"
  elif command -v xdg-open >/dev/null 2>&1; then xdg-open "$1"
  fi
}

# Opens $1 in the browser as soon as it answers; runs in the background so it
# works both in the foreground (exec) and detached paths.
open_when_ready() {
  [ "$OPEN" -eq 1 ] || return 0
  ( wait_for_url "$1" 120 && open_url "$1" ) >/dev/null 2>&1 &
}

# ------------------------------------------------------------ background ----
# Starts "$@" in its own session (own process group, no controlling terminal)
# so stop can signal the whole tree — npm -> concurrently -> tsx/vite — without
# ever touching the caller's process group. A plain `nohup cmd &` would share
# the group of whatever launched startup.sh (a terminal, an IDE task, CI).
start_bg() { # name, command...
  local name=$1 pid log; shift
  mkdir -p "$STATE_DIR"
  log="$STATE_DIR/$name.log"
  if command -v perl >/dev/null 2>&1; then
    perl -e 'use POSIX (); POSIX::setsid(); exec { $ARGV[0] } @ARGV or die "exec: $!"' -- "$@" >"$log" 2>&1 </dev/null &
  elif command -v python3 >/dev/null 2>&1; then
    python3 -c 'import os, sys
try:
    os.setsid()
except OSError:
    pass
os.execvp(sys.argv[1], sys.argv[1:])' "$@" >"$log" 2>&1 </dev/null &
  elif command -v setsid >/dev/null 2>&1; then
    setsid "$@" >"$log" 2>&1 </dev/null &
  else
    nohup "$@" >"$log" 2>&1 </dev/null &
  fi
  pid=$!
  printf '%s\n' "$pid" >"$STATE_DIR/$name.pid"
  info "$name: started in background (pid $pid, log .startup/$name.log)"
}

stop_pidfile() {
  local pidf=$1 name pid pgid target i
  name="$(basename "$pidf" .pid)"
  read -r pid _ <"$pidf" || true
  rm -f "$pidf"
  if [ -z "${pid:-}" ] || ! kill -0 "$pid" 2>/dev/null; then info "$name: not running"; return 0; fi
  # Signal the whole group only when the recorded pid leads it (the normal
  # case). Otherwise it may share the caller's group, so signal the pid alone.
  pgid="$(ps -o pgid= -p "$pid" 2>/dev/null | tr -d ' ' || true)"
  if [ -n "$pgid" ] && [ "$pgid" = "$pid" ]; then target="-$pid"; else target="$pid"; fi
  info "$name: stopping (pid $pid)"
  kill -TERM -- "$target" 2>/dev/null || true
  for i in 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20; do
    kill -0 "$pid" 2>/dev/null || return 0
    sleep 0.5
  done
  warn "$name: did not exit in 10s, killing"
  kill -KILL -- "$target" 2>/dev/null || true
}

# ------------------------------------------------------------- bootstrap ----
lock_hash() {
  node -e "const c=require('crypto'),f=require('fs');process.stdout.write(c.createHash('sha256').update(f.readFileSync('package-lock.json')).digest('hex'))"
}

# npm ci rather than npm install: it honours package-lock.json exactly and
# never rewrites it (npm install does, under some npm versions). Re-run only
# when node_modules is missing or the lock file changed since the last run.
install_deps() {
  local stamp=node_modules/.startup-lock-hash want have=''
  want="$(lock_hash)"
  [ -f "$stamp" ] && have="$(cat "$stamp")"
  if [ -d node_modules ] && [ "$have" = "$want" ]; then
    info "Dependencies match package-lock.json"
    return 0
  fi
  info "Installing dependencies (npm ci)"
  npm ci --no-audit --no-fund
  printf '%s' "$want" >"$stamp"
}

ensure_env_file() {
  [ -f .env ] && return 0
  local key
  if command -v openssl >/dev/null 2>&1; then key="$(openssl rand -hex 32)"
  else key="$(od -An -N32 -tx1 /dev/urandom | tr -d ' \n')"; fi
  ( umask 077; printf 'ENCRYPTION_KEY=%s\nPORT=%s\n' "$key" "$API_PORT" >.env )
  info "Created .env with a new ENCRYPTION_KEY"
}

bootstrap() {
  if [ "$INSTALL" -eq 1 ]; then
    install_deps
    ensure_env_file
  elif [ ! -f .env ]; then
    warn ".env is missing and --no-install was given; the dev server will generate a throwaway encryption key"
  fi
}

banner() {
  echo
  printf '%sWall-E (%s)%s\n' "$c_bold" "$1" "$c_off"
  case "$1" in
    dev)
      printf '  Dashboard   http://localhost:%s   (Vite, hot reload)\n' "$UI_PORT"
      printf '  API         http://localhost:%s   (/api, /v1)\n' "$API_PORT" ;;
    *)
      printf '  Dashboard + API   http://localhost:%s\n' "$API_PORT" ;;
  esac
  [ "$LAN" -eq 1 ] && echo "  LAN access is on; only use this on a trusted network."
  echo "  First run: add provider keys on the Keys page; the unified API key is in its header."
  if [ "$DETACH" -eq 1 ] || [ "$MODE" = podman ] || [ "$MODE" = docker ]; then echo "  Stop with: ./startup.sh stop"; else echo "  Stop with: Ctrl+C"; fi
  echo
}

# ----------------------------------------------------------------- nativ ----
start_nativ() {
  [ "$WITH_NATIV" -eq 1 ] || return 0
  [ -f external/nativ/Makefile ] || die "external/nativ is empty. Run: git submodule update --init external/nativ"
  [ "$(uname)" = Darwin ] || die "Nativ is a macOS app and cannot run here"
  command -v xcodegen >/dev/null 2>&1 || die "Nativ needs xcodegen and Xcode: brew install xcodegen"
  info "Nativ: building and launching (make xcode-run); the first build takes several minutes"
  start_bg nativ make -C external/nativ xcode-run
  echo "  Nativ serves OpenAI-compatible local models; add it as a provider in Wall-E once it is up."
  echo "  ./startup.sh stop ends the build/launch step only; quit the app itself from Nativ."
}

# ----------------------------------------------------------------- modes ----
run_dev() {
  pick_node
  bootstrap
  require_free "$API_PORT" "$UI_PORT"
  local script=dev; [ "$LAN" -eq 1 ] && script=dev:lan
  start_nativ
  banner dev
  open_when_ready "http://localhost:$UI_PORT"
  if [ "$DETACH" -eq 1 ]; then
    start_bg dev npm run "$script"
    if wait_for_url "http://localhost:$API_PORT/api/ping" 90 && wait_for_url "http://localhost:$UI_PORT/" 90; then
      info "API and dashboard are answering."
    else
      warn "Services did not answer within 90s. Last log lines:"; tail -n 25 "$STATE_DIR/dev.log" >&2; exit 1
    fi
  else
    exec npm run "$script"
  fi
}

run_prod() {
  pick_node
  bootstrap
  require_free "$API_PORT"
  info "Building server, CLI and dashboard (npm run build)"
  npm run build
  start_nativ
  banner prod
  open_when_ready "http://localhost:$API_PORT"
  local -a envs=(NODE_ENV=production PORT="$API_PORT")
  [ "$LAN" -eq 1 ] && envs+=(HOST=0.0.0.0)
  if [ "$DETACH" -eq 1 ]; then
    start_bg server env "${envs[@]}" node server/dist/index.js
    if wait_for_url "http://localhost:$API_PORT/api/ping" 60; then
      info "Server is answering."
    else
      warn "Server did not answer within 60s. Last log lines:"; tail -n 25 "$STATE_DIR/server.log" >&2; exit 1
    fi
  else
    exec env "${envs[@]}" node server/dist/index.js
  fi
}

# ------------------------------------------------------------ containers ----
# Podman is the default engine (rootless, daemonless); docker stays available
# for machines without it. CONTAINER_ENGINE=podman|docker overrides detection.
ENGINE=''
COMPOSE=()
export PODMAN_COMPOSE_WARNING_LOGS=false   # silence "Executing external compose provider"

engine_pick() {
  if [ -n "${CONTAINER_ENGINE:-}" ]; then ENGINE="$CONTAINER_ENGINE"
  elif [ "$MODE" = docker ]; then ENGINE=docker
  elif command -v podman >/dev/null 2>&1; then ENGINE=podman
  elif command -v docker >/dev/null 2>&1; then ENGINE=docker
  else die "Neither podman nor docker is installed. Podman: brew install podman && podman machine init"
  fi
  command -v "$ENGINE" >/dev/null 2>&1 || die "$ENGINE is not installed"
}

engine_ready() { "$ENGINE" info >/dev/null 2>&1; }

# Brings the engine up: the podman machine (macOS/Windows VM) or Docker Desktop.
engine_start() {
  engine_ready && return 0
  local i=0
  case "$ENGINE" in
    podman)
      [ "$(uname)" != Linux ] || die "podman is installed but 'podman info' fails; check the rootless setup"
      if [ -n "$(podman machine list --format '{{.Name}}' 2>/dev/null)" ]; then
        info "Starting the podman machine"
        podman machine start
      else
        info "No podman machine yet; creating one (downloads a VM image once, takes a few minutes)"
        podman machine init
        podman machine start
      fi ;;
    docker)
      if [ "$(uname)" = Darwin ] && [ -d /Applications/Docker.app ]; then
        info "Starting Docker Desktop"
        open -a Docker
      else
        die "The Docker daemon is not running"
      fi ;;
  esac
  until engine_ready; do
    i=$((i + 1)); [ "$i" -ge 120 ] && die "$ENGINE did not become ready within 120s"
    sleep 1
  done
}

# Sets COMPOSE to the compose command for ENGINE; returns 1 when none exists.
compose_pick() {
  case "$ENGINE" in
    podman)
      if podman compose version >/dev/null 2>&1; then COMPOSE=(podman compose)
      elif command -v podman-compose >/dev/null 2>&1; then COMPOSE=(podman-compose)
      else return 1; fi ;;
    docker) COMPOSE=(docker compose) ;;
    *) return 1 ;;
  esac
}
compose() { "${COMPOSE[@]}" "$@"; }

run_containers() {
  engine_pick
  engine_start
  compose_pick || die "podman needs a compose provider: brew install podman-compose (or the docker-compose plugin)"
  ensure_env_file
  start_nativ
  info "${COMPOSE[*]} up -d --build  (builds this checkout with $ENGINE, not upstream's image)"
  if [ "$LAN" -eq 1 ]; then env HOST_BIND=0.0.0.0 PORT="$API_PORT" "${COMPOSE[@]}" up -d --build
  else env PORT="$API_PORT" "${COMPOSE[@]}" up -d --build; fi
  banner "$ENGINE"
  open_when_ready "http://localhost:$API_PORT"
  if wait_for_url "http://localhost:$API_PORT/api/ping" 90; then info "Container is answering."
  else warn "Container did not answer within 90s; check: ${COMPOSE[*]} logs -f"; exit 1; fi
}

# True when ENGINE is installed, up, and has compose containers for this project.
compose_project_running() {
  command -v "$ENGINE" >/dev/null 2>&1 || return 1
  engine_ready || return 1
  compose_pick || return 1
  [ -n "$(compose ps -q 2>/dev/null)" ]
}

do_stop() {
  local pidf stopped=0
  for pidf in "$STATE_DIR"/*.pid; do
    [ -f "$pidf" ] || continue
    stop_pidfile "$pidf"; stopped=1
  done
  for ENGINE in ${CONTAINER_ENGINE:-podman docker}; do
    if compose_project_running; then
      info "${COMPOSE[*]} down"
      compose down
      stopped=1
    fi
  done
  [ "$stopped" -eq 1 ] || info "Nothing to stop."
}

do_status() {
  local pidf name pid p
  for pidf in "$STATE_DIR"/*.pid; do
    [ -f "$pidf" ] || continue
    name="$(basename "$pidf" .pid)"; read -r pid _ <"$pidf" || true
    if [ -n "${pid:-}" ] && kill -0 "$pid" 2>/dev/null; then echo "$name: running (pid $pid, log .startup/$name.log)"
    else echo "$name: not running (stale pid file)"; fi
  done
  for p in "$API_PORT" "$UI_PORT"; do
    if port_busy "$p"; then echo "port $p: in use by $(port_owner "$p")"; else echo "port $p: free"; fi
  done
  for ENGINE in ${CONTAINER_ENGINE:-podman docker}; do
    if compose_project_running; then echo "containers ($ENGINE):"; compose ps; fi
  done
}

case "$MODE" in
  dev)    run_dev ;;
  prod)   run_prod ;;
  podman) run_containers ;;
  docker) run_containers ;;
  stop)   do_stop ;;
  status) do_status ;;
esac
