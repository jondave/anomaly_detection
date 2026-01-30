#!/usr/bin/env python3
"""
Convenience script to record training data for different scenarios.

Provides easy ways to record:
1. Normal operation baseline
2. Faulty operation (for comparison)
3. Different driving patterns
"""
import os
import sys
import argparse
import subprocess
from datetime import datetime


def run_recorder(scenario_name, output_dir='./recorded_data'):
    """Run the feature recorder with a scenario-specific output directory"""
    scenario_dir = os.path.join(output_dir, scenario_name)
    os.makedirs(scenario_dir, exist_ok=True)
    
    print(f"\n{'='*60}")
    print(f"Recording {scenario_name.upper()} DATA")
    print(f"{'='*60}")
    print(f"Output directory: {scenario_dir}")
    print("\nInstructions:")
    
    instructions = {
        'normal': [
            "1. Drive the robot in normal patterns:",
            "   - Forward/backward at various speeds",
            "   - Gentle turns and rotations",
            "   - Smooth acceleration/deceleration",
            "   - Typical operating conditions",
            "2. Record for at least 5-10 minutes",
            "3. Press Ctrl+C when finished"
        ],
        'aggressive': [
            "1. Drive the robot with aggressive movements:",
            "   - Fast acceleration/deceleration",
            "   - Sharp turns",
            "   - Maximum speeds",
            "   - Quick direction changes",
            "2. Record for at least 5-10 minutes",
            "3. Press Ctrl+C when finished"
        ],
        'obstacle': [
            "1. Drive the robot over/around obstacles:",
            "   - Different terrain if available",
            "   - Bumpy surfaces",
            "   - Inclines (if supported)",
            "   - Normal speed, varied paths",
            "2. Record for at least 5-10 minutes",
            "3. Press Ctrl+C when finished"
        ],
        'fault': [
            "1. Intentionally induce faults (if safe):",
            "   - Misaligned wheels",
            "   - Uneven terrain",
            "   - Slipping conditions",
            "   - Motor stress (if applicable)",
            "2. Record for 5-10 minutes",
            "3. Press Ctrl+C when finished",
            "",
            "⚠️  CAUTION: Only test with faults that won't damage the robot!"
        ]
    }
    
    if scenario_name in instructions:
        for line in instructions[scenario_name]:
            print(line)
    
    print(f"\nPress Enter to start recording...")
    input()
    
    # Run the recorder
    try:
        cmd = [sys.executable, 'record_features.py', scenario_dir]
        subprocess.run(cmd, check=False)
    except KeyboardInterrupt:
        print("\nRecording stopped.")
    
    print(f"✓ Data saved to: {scenario_dir}")


def main():
    parser = argparse.ArgumentParser(
        description='Record sensor data for anomaly detection training'
    )
    parser.add_argument(
        'scenario',
        nargs='?',
        default='normal',
        choices=['normal', 'aggressive', 'obstacle', 'fault', 'custom'],
        help='Type of scenario to record (default: normal)'
    )
    parser.add_argument(
        '--output',
        '-o',
        default='./recorded_data',
        help='Base output directory (default: ./recorded_data)'
    )
    parser.add_argument(
        '--custom-name',
        '-n',
        help='Custom scenario name (used with "custom" scenario)'
    )
    
    args = parser.parse_args()
    
    scenario = args.scenario
    if scenario == 'custom':
        if not args.custom_name:
            print("Error: --custom-name required when using 'custom' scenario")
            sys.exit(1)
        scenario = args.custom_name
    
    run_recorder(scenario, args.output)


if __name__ == '__main__':
    main()
