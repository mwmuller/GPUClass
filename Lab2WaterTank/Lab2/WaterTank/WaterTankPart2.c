#include "MKL25Z4.h"                    // Device header
#include "system_MKL25Z4.h"             // Keil::Device:Startup

#define NUM_LEDS (3)
#define NUM_SENSORS (2)

// PORTD GPIO Pins
#define LED_GREEN (0)
#define LED_BLUE (5)
#define LED_YELLOW (2)
#define SW_TANK (4)

#define MASK(x) (1 << (x)) // sets field when changing states

// Prototypes
short determineState(const short); // returns the nextstate
void o_portInit(void);  // Init the ports
void setLEDState(const uint32_t); // handles the LEDs. Accepts state
short setSensorTrigger(const short, const short); // sets the sensor triggers and determines state
void determineLED(const short); // determines the LEDs to light

static const uint8_t ledPins[] = {LED_BLUE, LED_GREEN, LED_YELLOW}; // array of led pins
static short filling; // is the tank filling?

enum tankState // denotes the sensor trigger states
	{
	semiFill=0, // ETS = 0, FLS = 0
	tankFull, // ETS = 0, FLS = 1
	empty, // ETS = 1, FLS = 0
	};
	
	// Bit position was determined by figure provided for lab
	enum sensorID // Sensor bit position
	{
		FLS = 0, // Full tank sensor = 1st bit
		ETS // Empty sensor = 2nd bit
	};
	
int main()
{
	o_portInit(); // init ports; leds and switch
	
	//short curState = empty; // blue light is on, state 
	uint8_t i = 0; // sensor trigger iterator
	filling = 1;
	short currentState = MASK(ETS); // Empty sensor is on by default
	setLEDState(MASK(LED_BLUE)); // blue is on, tank is filling
	static const short triggerCycle[NUM_SENSORS] = {ETS, FLS}; // order which sensors trigger
	while(1)
	{
		// Only on button press set next state
		if(PORTD->ISFR & MASK(SW_TANK))
		{
			// Cycles through the states of the machine based on 
			if(!filling && i > 0)
			{	
				i--; // reverse order of triggers
				currentState = setSensorTrigger(triggerCycle[i], currentState); // determine the sensor state
			}
			else if(filling && i < NUM_SENSORS)
			{
				currentState = setSensorTrigger(triggerCycle[i], currentState); // determine the sensor state
				i++; // increment order
			}
			determineLED(currentState); // determine leds 
			PORTD->ISFR &= 0xffffffff; // clear button flag
		}
	}
	
}

void o_portInit(void) // initializes the ports
{
	uint8_t i; // iterator
	SIM->SCGC5 |= SIM_SCGC5_PORTD_MASK; // init PORTD
	
	for(i = 0; i < NUM_LEDS; i++)	// clearing and setting led mux
	{
		PORTD->PCR[ledPins[i]] &= ~PORT_PCR_MUX_MASK; // clear mux
		PORTD->PCR[ledPins[i]] |= PORT_PCR_MUX(1); // set mux to gpio pin
		PTD->PDOR |= MASK(ledPins[i]); // controlling output register
		PTD->PDDR |= MASK(ledPins[i]); // setting direction register
	}
	
	// Set filling led
	PTD->PSOR = MASK(LED_BLUE); // setting output for blue led
	
	// Clear sensor ports
	PTD->PCOR = MASK(LED_GREEN);
	PTD->PCOR = MASK(LED_YELLOW);
	
	// Switch init
	PORTD->PCR[SW_TANK] &= ~(PORT_PCR_MUX_MASK | PORT_PCR_IRQC_MASK); // clear the MUX and IRQC
	PORTD->PCR[SW_TANK] |= (PORT_PCR_MUX(1) | PORT_PCR_IRQC(10) | PORT_PCR_PE_MASK | PORT_PCR_PS_MASK); // gpio, check falling edge, pullup resistor set
	
	// set Direction for switch
	PTD->PDDR &= (uint8_t)~MASK(SW_TANK); // set pin direction as input
}

short setSensorTrigger(const short sensorTrigger, const short sensorStates)
{
	// sensorTrigger vals = 0 -> OFS, 1 -> FLS, 2 -> EFS.
	short newSensorStates = sensorStates; // 100 is default
	switch(sensorTrigger)
	{
		case ETS: // 10 bit position
			if(filling)
			{
				newSensorStates &= (short)~MASK(ETS); // clear ETS from sensor states
			}
			else
			{
				newSensorStates |= (short)MASK(ETS); // If the take is empty. Set the motor led to "filling"
				filling = 1;
			}
			break;
		case FLS: // 01 bit 
			if(filling) 
			{
				newSensorStates |= (short)MASK(FLS); // Adding FLS sensor trigger to sensor states
				filling = 0;
			}
			else 
			{
				newSensorStates &= (short)~MASK(FLS); // clear FLS trigger
			}
			break;
		default:
			// do nothing
		break;
	}
	return newSensorStates;
}

void determineLED(const short curState)
{
	switch(curState) // use the bit positions of the sensors to determine state.
	{
		case empty:
			setLEDState(MASK(LED_BLUE)); // Set the blue LED on if the tank is empty
		break;
		case semiFill: // The tank is filling and not empty
			if(filling)
			{
			setLEDState(MASK(LED_BLUE) | MASK(LED_GREEN)); // Motor on and tank not empty
			}
			else
			{
				setLEDState(MASK(LED_GREEN)); // If the motor is off and the tank is not empty
			}
		break;
		case tankFull: // tank is full
			if(filling)
			{
			setLEDState(MASK(LED_BLUE) | MASK(LED_YELLOW)); // tank is full and filling
			}
			else
			{
				setLEDState(MASK(LED_YELLOW)); // tank is full and not filling
			}
		break;
		default: // default filling. 
			setLEDState(MASK(LED_BLUE)); 
		break;
	}
}

void setLEDState(const uint32_t ledIds)
{
	uint8_t x;
	for(x = 0; x < NUM_LEDS; x++) // clear all leds
	{
		PTD->PCOR = MASK(ledPins[x]);
	}
	PTD->PSOR = ledIds; // set new lights
}
