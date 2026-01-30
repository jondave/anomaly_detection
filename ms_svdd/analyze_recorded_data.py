#!/usr/bin/env python3
"""
Utility to analyze and visualize recorded features data.
"""
import os
import csv
import sys
from pathlib import Path


def analyze_csv(csv_path):
    """Analyze a recorded CSV file and print statistics"""
    if not os.path.exists(csv_path):
        print(f"Error: File not found: {csv_path}")
        return
    
    print(f"\n{'='*70}")
    print(f"Analyzing: {os.path.basename(csv_path)}")
    print(f"{'='*70}")
    
    # Read CSV and get statistics
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    
    if not rows:
        print("No data in CSV file")
        return
    
    num_rows = len(rows)
    
    # Get headers
    headers = reader.fieldnames
    print(f"\nTotal samples: {num_rows}")
    print(f"Total features: {len(headers)}")
    print(f"\nFeature Groups:")
    
    # Group features
    groups = {}
    for header in headers:
        if 'cmd_vel' in header:
            prefix = 'Command Velocity'
        elif 'imu' in header:
            prefix = 'IMU (Accelerometer/Gyro)'
        elif 'odom' in header:
            prefix = 'Odometry'
        elif 'ranger' in header:
            prefix = 'Ranger Status'
        else:
            prefix = 'Other'
        
        if prefix not in groups:
            groups[prefix] = []
        groups[prefix].append(header)
    
    for group, features in sorted(groups.items()):
        print(f"  {group}: {len(features)} features")
        for feat in sorted(features)[:3]:
            print(f"    - {feat}")
        if len(features) > 3:
            print(f"    ... and {len(features)-3} more")
    
    # Analyze ranger motor data
    print(f"\n{'Motor Analysis':^70}")
    print(f"{'-'*70}")
    
    first_row = rows[0]
    motors_found = set()
    
    for i in range(4):
        motor_key = f'ranger_motor{i}_rpm'
        if motor_key in first_row:
            motors_found.add(i)
    
    print(f"Motors detected: {sorted(motors_found)}")
    
    # Get motor statistics
    if motors_found:
        print("\nMotor Statistics (across all samples):")
        for motor_id in sorted(motors_found):
            rpms = []
            currents = []
            temps = []
            
            for row in rows:
                try:
                    rpm = float(row.get(f'ranger_motor{motor_id}_rpm', 0))
                    current = float(row.get(f'ranger_motor{motor_id}_current', 0))
                    temp = float(row.get(f'ranger_motor{motor_id}_driver_temp', 0))
                    rpms.append(rpm)
                    currents.append(current)
                    temps.append(temp)
                except:
                    pass
            
            if rpms:
                print(f"\n  Motor {motor_id}:")
                print(f"    RPM:        min={min(rpms):7.1f}  max={max(rpms):7.1f}  avg={sum(rpms)/len(rpms):7.1f}")
                print(f"    Current:    min={min(currents):7.2f}A max={max(currents):7.2f}A avg={sum(currents)/len(currents):7.2f}A")
                print(f"    Temp:       min={min(temps):7.1f}°C max={max(temps):7.1f}°C avg={sum(temps)/len(temps):7.1f}°C")
    
    # Battery analysis
    if 'ranger_battery_voltage' in first_row:
        voltages = []
        for row in rows:
            try:
                voltage = float(row.get('ranger_battery_voltage', 0))
                voltages.append(voltage)
            except:
                pass
        
        if voltages:
            print(f"\n{'Battery':^70}")
            print(f"  Voltage:    min={min(voltages):6.2f}V max={max(voltages):6.2f}V avg={sum(voltages)/len(voltages):6.2f}V")
    
    print(f"\n{'='*70}\n")


def list_recorded_data(base_dir='./recorded_data'):
    """List all recorded data files"""
    if not os.path.exists(base_dir):
        print(f"No recorded data found in {base_dir}")
        return
    
    csv_files = list(Path(base_dir).rglob('*.csv'))
    
    if not csv_files:
        print(f"No CSV files found in {base_dir}")
        return
    
    print(f"\n{'='*70}")
    print(f"Found {len(csv_files)} recorded data file(s)")
    print(f"{'='*70}\n")
    
    for csv_file in sorted(csv_files):
        print(f"  {csv_file.parent.name}/{csv_file.name}")
        analyze_csv(str(csv_file))


def main():
    if len(sys.argv) > 1:
        # Analyze specific file
        analyze_csv(sys.argv[1])
    else:
        # List and analyze all recorded data
        list_recorded_data()


if __name__ == '__main__':
    main()
