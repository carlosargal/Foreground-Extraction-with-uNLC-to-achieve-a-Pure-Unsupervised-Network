ViPER-GT tool was used for doing the annotations (http://viper-toolkit.sourceforge.net/docs/).

The bounding boxes annotations of humans are provided for 24 classes. 14 classes out of 24 belong to UCF101 and the remaining 10 classes are from UCF11. (Note that UCF11 is a subset of UCF101.)

The following is the list of 14 classes from UCF101. In their annotations files, there are 101 columns representing 101 action classes:

BasketballDunk   
CliffDiving      
CricketBowling   
Fencing          
FloorGymnastics  
IceDancing       
LongJump         
PoleVault        
RopeClimbing     
SalsaSpin        
SkateBoarding    
Skiing           
Skijet           
Surfing          




The 10 classes from UCF11 are provided below. In their annotations files, there are 11 columns representing 11 action classes of UCF11. (The action "Swing" was removed due to inconsistencies in the frame rates.)

Basketball
Biking
Diving
GolfSwing
HorseRiding
SoccerJuggling
TennisSwing
TrampolineJumping
VolleyballSpiking
WalkingWithDog




***Note that the action names in UCF101 and UCF11 are slightly different. Please use the following correspondences to find the UCF101 class corresponding to each UCF11 class:

basketball_shooting => Basketball
biking => Biking
diving => Diving
golf_swing => GolfSwing
horse_riding => HorseRiding
soccer_juggling => SoccerJuggling
tennis_swing => TennisSwing
trampoline_jumping => TrampolineJumping
volleyball_spiking => VolleyballSpiking
walking => WalkingWithDog
