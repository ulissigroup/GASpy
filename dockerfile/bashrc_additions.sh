
# Double-tapping tab now lists all appropriate files
bind TAB:menu-complete

# Make the terminal & ls more colorful
export PS1="\[\033[36m\]\u\[\033[m\]@\[\033[32m\]\h:\[\033[33;1m\]\w\[\033[m\]\$ "
export CLICOLOR=1

# Always use long format and show directories when using `ls`
alias ls='ls -pl --color=auto'

# Make a new command to change directory and then list the files in it
function cs () {
    cd "$@" && ls
    }
