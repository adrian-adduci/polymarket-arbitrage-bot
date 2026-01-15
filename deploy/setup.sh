#!/bin/bash
#
# VPS Setup Script for Polymarket Arbitrage Bot
#
# Usage:
#   sudo bash deploy/setup.sh
#
# This script:
#   1. Creates a dedicated user (polybot)
#   2. Sets up directories with proper permissions
#   3. Creates Python virtual environment
#   4. Installs dependencies
#   5. Configures systemd service
#   6. Sets up log rotation
#
set -e

# Configuration
BOT_DIR="/opt/polymarket-bot"
LOG_DIR="/var/log/polymarket-bot"
DATA_DIR="/opt/polymarket-bot/data"
BOT_USER="polybot"
BOT_GROUP="polybot"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}=== Polymarket Bot VPS Setup ===${NC}"
echo ""

# Check if running as root
if [ "$EUID" -ne 0 ]; then
    echo -e "${RED}Error: Please run as root (sudo bash deploy/setup.sh)${NC}"
    exit 1
fi

# Check if script is run from the correct directory
if [ ! -f "requirements.txt" ]; then
    echo -e "${RED}Error: Please run this script from the project root directory${NC}"
    exit 1
fi

# 1. Create dedicated user
echo -e "${YELLOW}[1/7] Creating system user: $BOT_USER${NC}"
if ! id "$BOT_USER" &>/dev/null; then
    useradd -r -s /bin/false -d $BOT_DIR $BOT_USER
    echo "  Created user: $BOT_USER"
else
    echo "  User already exists: $BOT_USER"
fi

# 2. Create directories
echo -e "${YELLOW}[2/7] Creating directories${NC}"
mkdir -p $BOT_DIR
mkdir -p $LOG_DIR
mkdir -p $DATA_DIR
echo "  Created: $BOT_DIR"
echo "  Created: $LOG_DIR"
echo "  Created: $DATA_DIR"

# 3. Copy project files
echo -e "${YELLOW}[3/7] Copying project files${NC}"
cp -r . $BOT_DIR/
rm -rf $BOT_DIR/.git  # Remove git history
rm -f $BOT_DIR/.env   # Remove any existing .env (will be configured separately)
echo "  Copied project files to $BOT_DIR"

# 4. Set permissions
echo -e "${YELLOW}[4/7] Setting permissions${NC}"
chown -R $BOT_USER:$BOT_GROUP $BOT_DIR
chown -R $BOT_USER:$BOT_GROUP $LOG_DIR
chmod 750 $BOT_DIR
chmod 755 $LOG_DIR
chmod 750 $DATA_DIR
echo "  Set ownership to $BOT_USER:$BOT_GROUP"

# 5. Create virtual environment and install dependencies
echo -e "${YELLOW}[5/7] Setting up Python environment${NC}"
if [ ! -d "$BOT_DIR/venv" ]; then
    sudo -u $BOT_USER python3 -m venv $BOT_DIR/venv
    echo "  Created virtual environment"
fi
sudo -u $BOT_USER $BOT_DIR/venv/bin/pip install --upgrade pip -q
sudo -u $BOT_USER $BOT_DIR/venv/bin/pip install -r $BOT_DIR/requirements.txt -q
echo "  Installed dependencies"

# 6. Install systemd service
echo -e "${YELLOW}[6/7] Installing systemd service${NC}"
cp $BOT_DIR/deploy/polymarket-bot.service /etc/systemd/system/
systemctl daemon-reload
echo "  Installed systemd service"

# 7. Configure log rotation
echo -e "${YELLOW}[7/7] Configuring log rotation${NC}"
cat > /etc/logrotate.d/polymarket-bot << 'EOF'
/var/log/polymarket-bot/*.log {
    daily
    rotate 7
    compress
    delaycompress
    missingok
    notifempty
    create 0644 polybot polybot
    postrotate
        systemctl reload polymarket-bot 2>/dev/null || true
    endscript
}
EOF
echo "  Configured log rotation"

# Done
echo ""
echo -e "${GREEN}=== Setup Complete ===${NC}"
echo ""
echo -e "Next steps:"
echo -e "  1. ${YELLOW}Copy your .env file:${NC}"
echo -e "     sudo cp /path/to/your/.env $BOT_DIR/.env"
echo ""
echo -e "  2. ${YELLOW}Secure the .env file:${NC}"
echo -e "     sudo chmod 600 $BOT_DIR/.env"
echo -e "     sudo chown $BOT_USER:$BOT_GROUP $BOT_DIR/.env"
echo ""
echo -e "  3. ${YELLOW}Test in dry-run mode:${NC}"
echo -e "     sudo -u $BOT_USER $BOT_DIR/venv/bin/python $BOT_DIR/apps/dutch_book_runner.py --dry-run"
echo ""
echo -e "  4. ${YELLOW}Enable and start the service:${NC}"
echo -e "     sudo systemctl enable polymarket-bot"
echo -e "     sudo systemctl start polymarket-bot"
echo ""
echo -e "  5. ${YELLOW}Check status:${NC}"
echo -e "     sudo systemctl status polymarket-bot"
echo -e "     sudo journalctl -u polymarket-bot -f"
echo ""
echo -e "  6. ${YELLOW}Check health endpoint:${NC}"
echo -e "     curl http://localhost:8080/health"
echo ""
