#include "MKL25Z4.h"                    // Device header
#include "system_MKL25Z4.h"             // Keil::Device:Startup

// PORTD GPIO Pins
#define LED_RED (2)
#define LED_GREEN (0)
#define LED_BLUE (5)
#define LED_YELLOW (13)

#define MASK (x) (1ul << (x))

short handleState(const short);
short getSensor

enum tankState // defines the states for rthe tank state machine
	{
	empty=0,
	semiFill,
	tankFull,
	overflow
	};
	
int main()
{
	short curState = empty; // blue light is on, state 
	uint8_t waterLevel = 0; // Determines the amount of water in the tank
	short sensorTrigger = 0; // empty default
	while(1)
	{
		curState = handleState(curState);
		waterLevel += 10;
		
	}
	
}

short handleState(const short curState, short sensorTriggers)
{
	short nextState = 0;
	
		switch(curState)
		{
			case empty:
				nextState = (curState | sensorTriggers);
			break;
			case semiFill:
				nextState |= tankFull;
			break;
			case tankFull:
				nextState |=
			
		}
}
