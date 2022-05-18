# Introduction
This tutorial tries to take you through setup and using an RPi from scratch. All you need is that you've bought an RPi and have an internet connection. Actually, I recommend reading this if you're thinking of buying one, you'll see some instructions on what you need to buy so it's a good idea. Also the [RPi documentation][1] is the single best resource for starting with your RPi IMO, so you can also read that first. My post is somewhat complementary of that documentation.

## Introduction to the RPi
I've often come across the question, when do I need an RPi? You'll often need an RPi if you need a decent amount of portable processing. This is usually for complicated tasks like Machine Learning, Image Processing, running a server, etc. I've had mentees use RPi for various combinations of these too.

## What's so great about an RPi?
RPi fulfils the requirement of portable, high processing power. You can load an OS onto the system. It truly is the credit-card sized mini-computer. Added bonus is lots of support in the form of an active online community. RPi supports multiple popular operating systems too (eg. Ubuntu, Debian). This extends to support for multiple programming languages, which support operations on GPIO of RPi including Python and C/C++. With these, there are multiple plugin-like attachments, incuding a camera(RaspiCam). Support for multiple USB devices like a keyboard/mouse too!

## Setting Up
### Requirements
[You should read this page before buying anything.][2] Even if you don't have everything listed below you can make do (for eg WiFi Router/Ethernet Cable)

**Basic Setup**:RPi, MicroUSB attachment(to power RPi), a microSD card(below 64GB, recommended 16GB), MicroSD Card Reader, a WiFi Router and an Ethernet Cable(if possible).

**Easy Setup**: Basic setup + HDMI Cable, Monitor with HDMI support, Keyboard+Mouse(Could do with one of the two, not very sure but prefer having both).


### First Steps
First thing to do is choose an OS. If you're reading this, the most logical choice for you is the Raspbian OS (It's pretty nice, somewhat light GUI, most tutorials you find anywhere will work on Raspbian). Note that, Raspbian with Pixel DE(Desktop Environment) is the one that packs the User Interface, the other has only a terminal. If you are a complete n00b maybe you should download NOOBS (I'm kidding still use Raspbian). We start with burning an image file on the SD card. Crystal clear instructions are given at [this link][3].

### Connecting your RPi to a local network
So we have an OS on the RPi. Now what? We need to setup some sort of connection with the RPi. So here are your options:

***Easy setup:*** You should be super comfortable. You can just connect your RPi with an HDMI cable to the monitor, add a keyboard+mouse, and there you're done. Connect a good microUSB cable, insert the SD Card and switch the RPi's power on. You should note you can face issues with dealt with <a href="#bootup_issue">here</a>. It should show you something like [this][4]. Now you can use it like an OS! Connect to WiFi, scan the raspi-config for settings so you can setup whatever is appropriate for your case(don't worry it sounds intimidating but its super easy - when in doubt you can just start exploring options in the menu. There are some games in there which aren't super fun to play because of all the lag but you can try if you want! :) )
I think you're good from now on!

***Basic setup:*** No simple way to say this but I've failed at this method a few times. Sometimes it's because of the terrible WiFi module on the RPi(or terrible WiFi I use), sometimes I mess up, sometimes its a plain old fashioned error I can't understand how to debug because I can't see anything. Easiest thing to do would be use the easy setup once, configure your RPi, then it'll be fine to just use remote access. But if you can't access equipment, here we go!

 The crux is to get the RPi on the same local network as the PC. We can just use an ethernet cable and connect our RPi to our PCs. That is quite easy. If you don't have one I'll tell you what to do, keep following the post.
 
 Start with your SD Card (plugged in to your PC). We need to edit some specific files in it to make the RPi start an SSH server on boot (IMPORTANT). SSH will enable access to your RPi remotely. You can add edit the file `/etc/rc.local` and add the following line at the end
{% highlight bash %}
/etc/init.d/ssh start
{% endhighlight %}
 So rc.local is the "what-do-I-do-on-bootup" file. The code above starts an SSH server.

 Alternatively, SSH can be enabled by placing a file called `ssh` in to the boot folder. This flags the Pi to enable the SSH system on the next boot. Make a blank file and name it ssh. Place it there, that's it. HOWEVER, I DON'T RECOMMEND THIS. Simply because this file deletes on each boot(I think), so if you have to try booting twice maybe thrice, then this will kill. 

 If you have an ethernet cable you're done. If not, you need to setup the RPi to connect to your wireless network on it's own (because without an ethernet cable the RPi is not on a local network with you yet). ALTHOUGH, it's better to do this step for both with/without ethernet since there is no harm in getting this to work. 
 
 I found a decent blog post for setting up the network [here][5]. I'm not leaving you out to dry yet don't worry. BUT READ THAT LINK FIRST.
 
 Edit: Found an [even better one in RPi's documentation][6].

 <a name="file_edits_"></a> So what you can do is edit this one file at `/etc/wpa_supplicant/wpa_supplicant.conf` and what you need to do is copy this part:
{% highlight bash %}
country=in
update_config=1
ctrl_interface=/var/run/wpa_supplicant

network={
ssid="replace_with_your_ssid"
psk="replace_with_your_password"
key_mgmt=WPA-PSK
}
{% endhighlight %}

Since you are most probably using a WPA-PSK, this shouldn't be a problem, it should work for all WiFis in general. If you're not, [this][6] page can help you. If you don't think that's enough, [then try this page][7]. Don't let it intimidate you, just find out the kind of network you have. You can even use your phone, try connecting to your WiFi and check the options it needs you to enter. Search these options and you should get a good lead on what kind of network your WiFi is. This is the hardest part but after this things should be breezy.

Now all you gotta do is connect to the same WiFi from your PC!

Okay, for people with an Ethernet cable who couldn't make this work, there is some hope for you. You can connect to WiFi using your RPi terminal, but we'll get to that shortly.

### SSHing into the RPi

***Easy setup:*** You can open the terminal and type `sudo raspi-config`. This will show you some settings where you may need to change the locale and ***keyboard layout***. This is important because the layout can cause minor problems like in typing out passwords which you won't be able to debug for a long time (happened to me).

***Basic setup:*** Now that your RPi's SSH is on, you can place the SD Card into the RPi, and allow it to boot by connecting a power cord. Some (read most) people may face this issue of RPi not booting up properly or at all. <a name="bootup_issue"></a>[Please read this answer in full to resolve your issues, once and for all.][8] Since phones are getting MicroUSB cords with better power every year, this may become redundant in the near future. After bootup fire up your SSH client: in linux/Mac you have an inbuilt one, in Windows you have to install Putty.

Working with Windows its better to follow the [RPi docs][9], which comes packaged with instructions on getting your [PI's IP address][10]. I use nmap for getting the IP generally. I have had many troubles trying to use `raspberrypi.local`, so I don't recommend that.

### Working on your RPi
Working on your RPi isn't very difficult. I have tested a few ways and often for me, the way to go is to install `git` on your RPi and push changes online, while pulling changes into your RPi repo. For small changes, you can use nano/vim when SSHing into your RPi, and change it there. If you are looking for a `git` or a `vim` tutorial, you can find one at [Grundy, the WnCC Wiki of IIT Bombay][11].

And this is it! You can now work on your RPi from your PC and hopefully use it in your projects! There are some useful things you can also lookup, like triggering a script on boot [here][12] & [here][13], [building lighter OSes for RPi][14] or  using Raspbian(GUI) without the frizz, by [installing PixelDE on the lite version][15] (avoids installation of lots of software you probably don't need), [using RaspiCam and OpenCV with your RPi][16] and [doing some GPIO programming on Python][17]. 

Time to make your own projects!

[1]: https://www.raspberrypi.org/documentation/
[2]: https://www.raspberrypi.org/documentation/setup/
[3]: https://www.raspberrypi.org/documentation/installation/
[4]: https://youtu.be/Th_3AvK-EbM?t=273
[5]: https://howchoo.com/g/ndy1zte2yjn/how-to-set-up-wifi-on-your-raspberry-pi-without-ethernet
[6]: https://www.raspberrypi.org/documentation/configuration/wireless/wireless-cli.md
[7]: https://wiki.archlinux.org/index.php/WPA_supplicant
[8]: https://raspberrypi.stackexchange.com/questions/51615/raspberry-pi-power-limitations
[9]: https://www.raspberrypi.org/documentation/remote-access/ssh/windows.md
[10]: https://www.raspberrypi.org/documentation/remote-access/ip-address.md
[11]: https://www.wncc-iitb.org/wiki/index.php/The_Web_and_Coding_Club
[12]: https://www.raspberrypi-spy.co.uk/2015/02/how-to-autorun-a-python-script-on-raspberry-pi-boot/
[13]: https://www.dexterindustries.com/howto/run-a-program-on-your-raspberry-pi-at-startup/
[14]: https://www.raspberrypi.org/forums/viewtopic.php?t=133691
[15]: https://gist.github.com/kmpm/8e535a12a45a32f6d36cf26c7c6cef51
[16]: https://www.pyimagesearch.com/2015/03/30/accessing-the-raspberry-pi-camera-with-opencv-and-python/
[17]: https://pythonprogramming.net/gpio-raspberry-pi-tutorials/
