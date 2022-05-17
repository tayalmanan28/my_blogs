---
toc: true
layout: post
description: A Blog to explain how to choose Batteries for Robotics.
image: images/battery.png
categories: [Battery,Robotics]
title: How to Choose Batteries
---

Batteries are the energy storage units of many devices that we come across every day; they are available in different forms, sizes, parameters, and shapes. You can commonly find them being used in automotive, Backup power supplies, mobile devices, laptops, iPads, and many other portable electronic devices. But, not all the devices can use the same kind of battery; each and every device has its own specifications and power supply requirements and you will need a battery selection guide to pick the right battery for your application. So in this article will look into the factors to consider while selecting a battery for your next electronic product design. If you are completely news to batteries then it is recommended to read this article on types of batteries and their applications to understand the basics of battery before you proceed further.

# Factors for Choosing Batteries
While choosing a battery for your application you must know about the important parameters involved in its operation. The reality about the battery is that there is no common type of battery for all the applications since no battery is perfect. If you want to utilize one parameter of the battery you should be able to handle the depletion of other parameters. For example, if you want your battery to deliver lots of power for your application, the cell internal resistance should be minimized which is only possible by increasing the electrode surface area. This also increases inactive components such as current collectors and conductive aid, so energy density is traded off to gain power. In order to provide exactly what you want in your application, you must give up something to gain the other in a battery. The important battery parameters are given in the following image.


Now, let’s look into each battery parameter briefly to understand its importance and impact on battery performance during operation.

## Rechargeable / Non-Rechargeable batteries
There might not be much confusion in choosing between a primary and secondary battery, you must only know if you want the battery to be used once or multiple times. The primary (Non-Rechargeable) battery can be used for occasional uses like toys, Flashlights, Smoke alarm, etc. They are also used in devices in which charging isn’t possible like pacemakers, wristwatches and hearing aids. The Secondary (rechargeable) batteries can be used in the applications where there is a need for a regular power source like mobile phones, laptops, Automotives, etc. The secondary batteries always have a higher self-discharge rate compared to primary batteries which are an ignorant fact due to its ability to be recharged.

 

## Availability of Space
The Batteries are available in various shapes and size like button cells, cylindrical cells, Pouch cells & prismatic cells. The size of the battery really matters in order to make your device easily portable. The standard sizes available are AA, AAA and 9V batteries suitable for portable devices. Commonly lithium batteries (pouch type) are preferred in applications where there is less space but more power requirement. If the power requirement is less then coin cells can also be considered since they are very compact and the smallest of battery types.

# Different Shapes of Battery

 

## System Operating Voltage
The battery voltage is one of the most important characteristics of the battery, which is determined based on the electrode & electrolyte used (Chemical Reaction). There is a common misconception that a fully discharged battery will have 0V it is clearly not the case in any battery. In fact, if a battery reads 0V then it probably is dead. The output voltage of a battery should always read between its nominal voltage level.

 

The Zinc-Carbon battery and Nickel-metal hydride battery uses water as an electrolyte and delivers a nominal voltage of 1.2V to 2V, whereas the lithium-based batteries use organic electrolytes that can deliver a nominal voltage of 3.2 to 4V. Most of the electronic pieces of equipment operate in the voltage range of 3V. If you use a lithium-based battery a single cell battery will be enough to operate the equipment. Do remember that the voltage of the battery will not be stable and will vary between a minimum value and maximum value based on the available capacity in the battery. This minimum and maximum value of each battery is shown below.

## Minimum and Maximum Value of Battery

If your circuit is operating at 5V and you are powering it with a lithium battery, then your nominal voltage will only be 3.2V to 4V. In these cases, boost converter circuits are used to convert the battery voltage to 5V required for the circuit. If your operating voltage is very high like 24V or 12V then you can either use a 12V lead-acid battery or if you need high power density then you can combine more than one lithium cells in series to increase the resulting output voltage. 

 

## Operating Temperature
The battery performance can be dramatically changed by the temperature, for instance the battery that is operating with aqueous electrolytes cannot be used in temperature conditions below 0°C as they aqueous electrolyte might get frozen under 0°C, in the same way, the lithium-based batteries might operate up to -40°C but the performance might be dropped.

## Battery Performance

The lithium-ion batteries have the maximum charging rate between the temperature ranges of about 20°C to 45°C. If you want to charge beyond this temperature range lower current/voltage need to be used, this will result in longer charging time. If the temperature drops below 5°C or 10°C lithium dendrite plating will be formed in the electrolyte which needs to be prevented by trickle charge.

 

## Capacity of the battery - Power & Energy
The power of the battery determines the runtime of a battery. The power/Capacity of the battery is expressed in Watt-hours (Wh). The Watt-hour is calculated by multiplying the battery voltage (V) by the amount of current that a battery can deliver for a certain amount of time. The voltage of the battery is almost fixed and the current that a battery can deliver is printed on the battery, expressed in Ampere-hour rating (Ah or mAh).

Consider a battery of 5V with 2 Amp-hour (Ah) capacity, hence it has a power of 10Wh. A battery with the capacity of 2Ah can deliver 2 Amp for one hour or 0.2A for 10 hours or 0.02A (20mA) for 100 hours. Battery manufacturers always specify the capacity at a given discharge rate, temperature, and cut-off voltage, where the capacity always depends on all three factors.

 

The capacity of a battery will tell us how much power it can deliver to an application. For example, consider a 12V, 10Ah car battery, the actual capacity of the battery is 120Wh (12V x 10Ah), but in a laptop battery of 3.6V that has the same 10Ah dissipation will have a capacity of 36Wh (3.6Vx 10Ah). From the example you can see even they have the same Ah the amount of power that a car battery can store is three times higher than a laptop battery.

The following picture will give you more clarity about how the battery capacity differs in different types of batteries.

# Capacity of Batteries

Batteries with high power always provide a faster discharge capability at high drain rates like power tools or automobile starter battery applications, most of the high power batteries will have a low energy density.

 

# Battery Chemistry
By this time you would have understood that all the properties of a battery are always depending on the chemistry involved in the battery, so you should be more conscious while you choose the type of battery. On the basis of the chemistry involved in the operation, batteries are classified as Lead Acid Batteries, Alkaline Batteries, Ni-Cad Batteries (Nickel Cadmium), Ni- MH Batteries (Nickel Metal Hydride), Li-Ion (Lithium-Ion) and LiPoly (Lithium Polymer) Batteries

# Different types of Battery

 

## Cost of Battery
In most portable electronics products the battery will be one among the expensive item in the Bill of Materials (BOM), hence most of the time it will affect the overall cost of your electronic applications. Hence, you should know your needs and budget of your product and then choose the right battery for your product.

 

## Shelf Life
Not all the batteries are used immediately after manufacturing, they stay on the shelf for a long time before it’s being used. The shelf life of a battery tells you how long a battery can be kept unused. The Shelf life is mostly considered as a fact in primary batteries only as the secondary batteries can be recharged whenever they are used. For example in a fire alarm siren system, the battery might sit there idle for years before it detects a fire and triggers the alarm. So care should be taken that the battery retains its performance even if it is kept unused for a long time.

 

## Which battery should I choose?
Now that we have looked into the parameters you should consider before choosing the battery for a portable electronic application, let’s look into the common cases of choosing the battery. Note that these are just tips and not hard written rules.

- For products that consume more power like projectors, large sound systems, and motorized projects you should use lead-acid batteries. If you are going to have heavy usage of the battery you should go for ‘Marine deep cycle’ batteries.
- If your electronics need to be super small like an inch on each side you should go for the lithium coin cells or little lithium polymer cells.
- If you are going to produce the component in large quantity use inexpensive alkaline batteries of popular sizes. So the customer finds it easy to replace them.
- If you want the device to be user-serviceable, like the users can change the battery by themselves go for 9V or AA-size batteries.
- Use 3 Alkaline (4.5V) or 4NiMH cells (4.8V) if the circuit needs approximately 5V input.
- To build a rechargeable battery pack use a battery holder from your local shop and stick it with NiMH batteries and then start recharging your battery.
- If you want to replace your alkaline battery with any of the rechargeable batteries, test your device to make sure that it can operate at lower voltage without any issue.
- If you want your battery to have a longer life span always use a high-quality charger with sensors to maintain proper charging and trickle charging because using a cheap charger will kill off your cells in the battery pack.
