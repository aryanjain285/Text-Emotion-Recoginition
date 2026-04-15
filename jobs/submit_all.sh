#!/bin/bash
# ==============================================================
# submit_all.sh — Submit all jobs with correct dependencies.
# Steps 1, 2, 3, 3b run in parallel. Step 4 waits for all.
#
# Usage:
#   cd /home/users/$USER/Text-Emotion-Recognition
#   bash jobs/submit_all.sh
# ==============================================================

echo "Submitting experiment jobs..."

# Steps 1, 2, 3, 3b can run in parallel (independent)
JOB1=$(qsub jobs/job_step1.pbs)
echo "Step 1 (baselines):       $JOB1"

JOB2=$(qsub jobs/job_step2.pbs)
echo "Step 2 (BERT models):     $JOB2"

JOB3=$(qsub jobs/job_step3.pbs)
echo "Step 3 (ablation):        $JOB3"

JOB3B=$(qsub jobs/job_step3b.pbs)
echo "Step 3b (kernel analysis): $JOB3B"

# Extract job IDs for dependency
ID1=$(echo $JOB1 | cut -d. -f1)
ID2=$(echo $JOB2 | cut -d. -f1)
ID3=$(echo $JOB3 | cut -d. -f1)
ID3B=$(echo $JOB3B | cut -d. -f1)

# Step 4 depends on all previous steps completing
JOB4=$(qsub -W depend=afterok:${ID1}:${ID2}:${ID3}:${ID3B} jobs/job_step4.pbs)
echo "Step 4 (visualize):       $JOB4 (waits for steps 1-3b)"

echo ""
echo "All jobs submitted. Monitor with: qstat -u $USER"
echo ""
echo "OR submit everything as a single 6-hour job:"
echo "  qsub jobs/job_all.pbs"
