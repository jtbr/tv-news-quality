#!/bin/bash
#remove ascii characters 0-12 from the text
LANG=ISO8895-1 sed -i 's/[\x00-\x0c]//g' $1