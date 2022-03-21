#include "MKL25Z4.h"                    // Device header
#include "RTE_Components.h"             // Component selection

// PORTC pins 7 segment
#define SEG_A (7)
#define SEG_B (3)
#define SEG_C (6)
#define SEG_D (10)
#define SEG_E (5)
#define SEG_F (0)
#define SEG_G (4)

// number of segements
#define SEG_NUM (7)

// PORTC pins Switches
// SW_n indicates bit position n
#define SW_0 (17)
#define SW_1 (16)
#define SW_2 (13)
#define SW_3 (12)

// num of switches
#define SW_NUM (4)

#define MASK(x) (uint32_t)(1UL << (uint32_t)x)


void o_initPins(void);
void handleSw(void);
void outputSeg7(uint8_t);
void setLedMask(uint8_t);

// display bits to be used in or'ing
enum segState
{				// output bits -- Input Switches
	zero = 0, 		// 1111110 -- 0000
	one,			// 0110000 -- 0001
	two, 			// 1101101 -- 0010
	three,			// 1111001 -- 0011
	four,			// 0110011 -- 0100
	five,			// 1011011 -- 0101
	six,			// 1011111 -- 0110
	seven,			// 1110000 -- 0111
	eight,			// 1111111 -- 1000
	nine,			// 1111011 -- 1001
	ten,			// 1110111 -- 1010 (a)
	eleven,   		// 0011111 -- 1011 (b)
	twelve,			// 1001111 -- 1100 (c)
	thirteen,		// 0111101 -- 1101 (d)
	fourteen,		// 1001111 -- 1110 (e)
	fifteen   		// 1000111 -- 1111 (f)
};

static const short Segments[7] = {SEG_A, SEG_B, SEG_C, SEG_D, SEG_E, SEG_F, SEG_G};

// In order from LSB -> MSB
static const short Switches[4] = {SW_0, SW_1, SW_2, SW_3};

// determine if the state of the display has changed
static short prev_state = 0;

int main()
{
	// Init the pins for Port c
	o_initPins();
	PTC->PSOR |= MASK(SEG_G);
	while(1)
	{
		// handles the switches that are pressed
		handleSw();
	}
}

void o_initPins()
{
	uint8_t i; // iterator
	SIM->SCGC5 |= SIM_SCGC5_PORTC_MASK;
	
	for(i = 0; i < SEG_NUM; i++)
	{
		// sanatize and set GPIO pin output
		PORTC->PCR[Segments[i]] &= ~PORT_PCR_MUX_MASK;
		PORTC->PCR[Segments[i]] |= PORT_PCR_MUX(1);
			
		PTC->PDDR |= MASK(Segments[i]);
		PTC->PDDR |= MASK(Segments[i]); // setting direction register
		PTC->PCOR |= MASK(Segments[i]);
	}
	for(i = 0; i < SW_NUM; i++)
	{
		// sanatize and enable pullup for switches
		PORTC->PCR[Switches[i]] &= ~PORT_PCR_MUX_MASK;
		PORTC->PCR[Switches[i]] |= PORT_PCR_MUX(1);
			
		PTC->PDDR &= ~MASK(Switches[i]);
	}
}

void handleSw()
{
	short switchesPressed = 0;
	uint8_t i; // iterator
	/*
	zero = 0, 	// 0111-1110 -- 0000
	one,		// 0011-0000 -- 0001
	two, 		// 0110-1101 -- 0010
	three,		// 0111-1001 -- 0011
	four,		// 0011-0011 -- 0100
	five,		// 0101-1011 -- 0101
	six,		// 0101-1111 -- 0110
	seven,		// 0111-0000 -- 0111
	eight,		// 0111-1111 -- 1000
	nine		// 0111-1011 -- 1001
	*/
	for(i = 0; i < SW_NUM; i++)
	{
		if(PTC->PDIR & MASK(Switches[i]))
		{
			switchesPressed |= MASK(i); // construct switches binary value
		}
	}
	if(prev_state != switchesPressed)
	{
		switch(switchesPressed)
		{
			case zero:
				setLedMask(0x7E); // 0111-1110 -- 0000
				break;
			case one:
				setLedMask(0x30); // 0011-0000 -- 0001
				break;
			case two:
				setLedMask(0x6D); // 0110-1101 -- 0010
				break;
			case three:
				setLedMask(0x79); // 0111-1001 -- 0011
				break;
			case four:
				setLedMask(0x33); // 0011-0011 -- 0100
				break;
			case five:
				setLedMask(0x5B); // 0101-1011 -- 0101
				break;
			case six:
				setLedMask(0x5F); // 0101-1111 -- 0110
				break;
			case seven:
				setLedMask(0x70); // 0111-0000 -- 0111
				break;
			case eight:
				setLedMask(0x7F); // 0111-1111 -- 1000
				break;
			case nine:
				setLedMask(0x7B); // 0111-1011 -- 1001
				break;
		  	case ten:
				setLedMask(0x77); // 0111-0111 -- 1010 (a)
				break;
			case eleven:
				setLedMask(0x1F); // 0001-1111 -- 1011 (b)
				break;
			case twelve:
				setLedMask(0x4E); // 0100-1110 -- 1100 (c)
				break;
			case thirteen:
				setLedMask(0x3D); // 0011-1101 -- 1101 (d)
				break;
			case fourteen:
				setLedMask(0x4F); // 0100-1111 -- 1110 (e)
				break;
			case fifteen:
				setLedMask(0x47); // 0100-0111 -- 1111 (f)
				break;
			default:
				setLedMask(0x7E); // 0111-1110 -- 0000
				break;
		}
	// If the state of the display has changed, update the disply
	}
	prev_state = switchesPressed;
}


void setLedMask(uint8_t arrPins)
{
	int i; // iterator
	// iterate through the bits and set that pin
	for(i = SEG_NUM - 1; i >= 0; i--)
	{
		if(1 & arrPins)
		{
			PTC->PCOR |= MASK(Segments[i]);
		}
		else
		{
			PTC->PSOR |= MASK(Segments[i]); // turn off all LEDs
		}
		arrPins = arrPins >> 1UL; // shift to the right and check next bit
	}
	
}

