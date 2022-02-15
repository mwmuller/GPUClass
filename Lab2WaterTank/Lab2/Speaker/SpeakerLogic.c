#include "MKL25Z4.h"                    // Device header
#include "RTE_Components.h"             // Component selection

#define SPKR_SHIFT (0)
#define HZDELAY (20000) // alternate frequency
#define MASK(x) (1ul << x)
#define SW_SWITCH (4)

void initSpeaker(void);
void delay(volatile unsigned int);
static const uint16_t freq[] = {100, 2500, 5000, 10000, 20000};
int main()
{
	initSpeaker(); // init speaker port
	short i = 0;
	while(1)
	{
		if(freq[i] >= 2500)
		{
			PTC->PTOR = MASK(SPKR_SHIFT);
			delay(freq[i]);
		}
		if(PORTD->ISFR & MASK(SW_SWITCH))
		{
			i = (i + 1) % 5;
			
			PORTD->ISFR &= 0xffffffff; // clear button flag
		}
		//delay(2500);
	}
}

void initSpeaker()
{
	SIM->SCGC5 |= (SIM_SCGC5_PORTC_MASK | SIM_SCGC5_PORTD_MASK);
	PORTC->PCR[SPKR_SHIFT] |= PORT_PCR_MUX(1);
	PORTD->PCR[SW_SWITCH] &= ~(PORT_PCR_MUX_MASK | PORT_PCR_IRQC_MASK);
	PORTD->PCR[SW_SWITCH] |= (PORT_PCR_MUX(1) | PORT_PCR_IRQC(10) | PORT_PCR_PE_MASK | PORT_PCR_PS_MASK); // gpio, check falling edge, pullup resistor set
	PTD->PDDR &= (uint8_t)~MASK(SW_SWITCH); // set pin direction as input
	PTC->PDDR |= MASK(SPKR_SHIFT);
	PTC->PDOR |= MASK(SPKR_SHIFT);
}

void delay(volatile unsigned int time_del)
{
	while(time_del--)
	{
		// do nothing
	}
}
