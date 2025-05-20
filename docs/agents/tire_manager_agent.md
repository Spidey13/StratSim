# Tire Manager Agent

## Overview

The TireManagerAgent is a critical component of the F1 Race Strategy Simulator responsible for tracking tire state and providing strategic recommendations for pit stops based on tire condition. It builds on the TireWearAgent's degradation modeling while adding state management and decision-making capabilities.

## Implementation Logic

### Core Concepts

- **Tire State Management**: The agent maintains the current state of tires (compound, age, wear percentage, grip level)
- **Performance Cliff Detection**: Identifies when tires approach their performance "cliff" (point of rapid degradation)
- **Pit Stop Recommendations**: Provides actionable guidance on when to pit and which compound to use next
- **Rules & Regulations**: Considers F1-specific rules like mandatory compound usage during races

### Tire Health Classification

The agent classifies tire health into five states based on wear percentage:

1. **New** (< 20% wear): Fresh tires with optimal performance
2. **Good** (20-80% of cliff threshold): Healthy tires with predictable grip
3. **Marginal** (80-100% of cliff threshold): Approaching performance cliff, consider pitting
4. **Critical** (at or just past cliff): Rapid performance loss, urgent pit recommendation
5. **Expired** (well past cliff): Severe degradation, emergency pit needed

### Compound-Specific Thresholds

Different tire compounds have different wear characteristics:

```
SOFT: 65% wear threshold (degrades quickly)
MEDIUM: 75% wear threshold 
HARD: 85% wear threshold (most durable)
INTERMEDIATE: 60% threshold (for mixed conditions)
WET: 40% threshold (degrades when track dries)
```

### Decision Making Process

When determining pit recommendations, the agent follows this logic:

1. **Wear Assessment**: Evaluates current tire wear against compound-specific thresholds
2. **Urgency Classification**: 
   - High urgency if tires are critical 
   - Medium urgency if tires are marginal
   - No urgency if tires are good/new
3. **Compound Selection**:
   - Weather-based selection (WET/INTERMEDIATE for rain)
   - Mandatory compound requirements (ensures regulatory compliance)
   - Optimal performance based on remaining race distance

### Integration with Other Agents

The TireManagerAgent:
- Uses the TireWearAgent to calculate wear progression
- Provides inputs to the StrategyAgent for pit stop decisions
- Receives race context from the RaceOrchestrator

## Testing and Visualization

The agent is tested with a race simulation that:
- Simulates a complete race with changing conditions
- Tracks tire state lap-by-lap
- Evaluates pit stop recommendations
- Records tire performance data

For visualization and analysis, we provide a script that generates:
1. **Tire Wear Progression**: Charts showing tire wear over time with cliff thresholds
2. **Grip Level Analysis**: Visualization of grip degradation 
3. **Race Overview**: Combined view of tire performance with weather conditions

Run the test and visualization with:
```bash
# First run the simulation
python scripts/model_testing/test_tire_manager_agent.py

# Then generate visualizations
python scripts/model_testing/visualize_tire_simulation.py
```

## Usage Example

```python
# Initialize tire manager
tire_manager = TireManagerAgent()

# Process each lap
result = tire_manager.process({
    "current_lap": current_lap,
    "circuit_id": "monza",
    "compound": "MEDIUM",  # Only needed on first lap or pit stops
    "is_pit_lap": False,
    "weather": weather_data,
    "laps_remaining": laps_remaining,
    "strategy": strategy_data
})

# Check results
print(f"Tire wear: {result['tire_wear']}%")
print(f"Grip level: {result['grip_level']}")

# Check if pit stop needed
if result["pit_recommendation"]["should_pit"]:
    next_compound = result["pit_recommendation"]["recommended_compound"]
    print(f"Pit recommended: Switch to {next_compound}")
```

## Future Improvements

1. **Machine Learning Integration**: Train a model on historical F1 data for more accurate wear predictions
2. **Driver-Specific Adjustments**: Account for driving style affecting tire wear rates
3. **Opponent Strategy Consideration**: Factor in what competitors are doing
4. **Undercut/Overcut Analysis**: Calculate optimal timing for strategic pit stops
5. **Safety Car/VSC Opportunistic Pitting**: Detect opportunities for "free" pit stops 