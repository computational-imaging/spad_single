#! /bin/bash
grep blacklisted nohup.out | awk '{print $2}' > blacklist.txt
