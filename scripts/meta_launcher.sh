
#!/usr/bin/env bash

# WARNING bash is including last step


for i in {1..1599}; do

    qsub perceptromic_${i}.sh

    done
