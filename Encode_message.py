# -*- coding: utf-8 -*-
"""
Created on Tue Jul 30 14:40:38 2019

@author: ziyad
"""

import numpy as np
import cv2
import matplotlib.pyplot as plt
from ae import get_image_to_encode
from LSB_Matching import LSB_Matching_encode, string_to_binary, LSB_Matching_decode


rows = 128
cols = 128
image = cv2.imread('images.jpeg')
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
image = cv2.resize(image, (rows,cols))
image = np.array(image)

string_to_encode = 'This is a message I would like to encode if possible!'

# clear visual distortions when using almost all the pixels for encoding
#string_to_encode = 'In the streets of Verona another brawl breaks out between the servants of the feuding noble families of Capulet and Montague. Benvolio, a Montague, tries to stop the fighting, but is himself embroiled when the rash Capulet, Tybalt, arrives on the scene. After citizens outraged by the constant violence beat back the warring factions, Prince Escalus, the ruler of Verona, attempts to prevent any further conflicts between the families by decreeing death for any individual who disturbs the peace in the future. Romeo, the son of Montague, runs into his cousin Benvolio, who had earlier seen Romeo moping in a grove of sycamores. After some prodding by Benvolio, Romeo confides that he is in love with Rosaline, a woman who does not return his affections. Benvolio counsels him to forget this woman and find another, more beautiful one, but Romeo remains despondent. Meanwhile, Paris, a kinsman of the Prince, seeks Juliet’s hand in marriage. Her father Capulet, though happy at the match, asks Paris to wait two years, since Juliet is not yet even fourteen. Capulet dispatches a servant with a list of people to invite to a masquerade and feast he traditionally holds. He invites Paris to the feast, hoping that Paris will begin to win Juliet’s heart. Romeo and Benvolio, still discussing Rosaline, encounter the Capulet servant bearing the list of invitations. Benvolio suggests that they attend, since that will allow Romeo to compare his beloved to other beautiful women of Verona. Romeo agrees to go with Benvolio to the feast, but only because Rosaline, whose name he reads on the list, will be there. In Capulet’s household, young Juliet talks with her mother, Lady Capulet, and her nurse about the possibility of marrying Paris. Juliet has not yet considered marriage, but agrees to look at Paris during the feast to see if she thinks she could fall in love with him. The feast begins. A melancholy Romeo follows Benvolio and their witty friend Mercutio to Capulet’s house. Once inside, Romeo sees Juliet from a distance and instantly falls in love with her; he forgets about Rosaline completely. As Romeo watches Juliet, entranced, a young Capulet, Tybalt, recognizes him, and is enraged that a Montague would sneak into a Capulet feast. He prepares to attack, but Capulet holds him back. Soon, Romeo speaks to Juliet, and the two experience a profound attraction. They kiss, not even knowing each other’s names. When he finds out from Juliet’s nurse that she is the daughter of Capulet—his family’s enemy—he becomes distraught. When Juliet learns that the young man she has just kissed is the son of Montague, she grows equally upset. As Mercutio and Benvolio leave the Capulet estate, Romeo leaps over the orchard wall into the garden, unable to leave Juliet behind. From his hiding place, he sees Juliet in a window above the orchard and hears her speak his name. He calls out to her, and they exchange vows of love. Romeo hurries to see his friend and confessor Friar Lawrence, who, though shocked at the sudden turn of Romeo’s heart, agrees to marry the young lovers in secret since he sees in their love the possibility of ending the age-old feud between Capulet and Montague. The following day, Romeo and Juliet meet at Friar Lawrence’s cell and are married. The Nurse, who is privy to the secret, procures a ladder, which Romeo will use to climb into Juliet’s window for their wedding night. The next day, Benvolio and Mercutio encounter Tybalt—Juliet’s cousin—who, still enraged that Romeo attended Capulet’s feast, has challenged Romeo to a duel. Romeo appears. Now Tybalt’s kinsman by marriage, Romeo begs the Capulet to hold off the duel until he understands why Romeo does not want to fight. Disgusted with this plea for peace, Mercutio says that he will fight Tybalt himself. The two begin to duel. Romeo tries to stop them by leaping between the combatants. Tybalt stabs Mercutio under Romeo’s arm, and Mercutio dies. Romeo, in a rage, kills Tybalt. Romeo flees from the scene. Soon after, the Prince declares him forever banished from Verona for his crime. Friar Lawrence arranges for Romeo to spend his wedding night with Juliet before he has to leave for Mantua the following morning.In her room, Juliet awaits the arrival of her new husband. The Nurse enters, and, after some confusion, tells Juliet that Romeo has killed Tybalt. Distraught, Juliet suddenly finds herself married to a man who has killed her kinsman. But she resettles herself, and realizes that her duty belongs with her love: to Romeo. Romeo sneaks into Juliet’s room that night, and at last they consummate their marriage and their love. Morning comes, and the lovers bid farewell, unsure when they will see each other again. Juliet learns that her father, affected by the recent events, now intends for her to marry Paris in just three days. Unsure of how to proceed—unable to reveal to her parents that she is married to Romeo, but unwilling to marry Paris now that she is Romeo’s wife—Juliet asks her nurse for advice. She counsels Juliet to proceed as if Romeo were dead and to marry Paris, who is a better match anyway. Disgusted with the Nurse’s disloyalty, Juliet disregards her advice and hurries to Friar Lawrence. He concocts a plan to reunite Juliet with Romeo in Mantua. The night before her wedding to Paris, Juliet must drink a potion that will make her appear to be dead. After she is laid to rest in the family’s crypt, the Friar and Romeo will secretly retrieve her, and she will be free to live with Romeo, away from their parents’ feuding.Juliet returns home to discover the wedding has been moved ahead one day, and she is to be married tomorrow. That night, Juliet drinks the potion, and the Nurse discovers her, apparently dead, the next morning. The Capulets grieve, and Juliet is entombed according to plan. But Friar Lawrence’s message explaining the plan to Romeo never reaches Mantua. Its bearer, Friar John, gets confined to a quarantined house. Romeo hears only that Juliet is dead. Romeo learns only of Juliet’s death and decides to kill himself rather than live without her. He buys a vial of poison from a reluctant Apothecary, then speeds back to Verona to take his own life at Juliet’s tomb. Outside the Capulet crypt, Romeo comes upon Paris, who is scattering flowers on Juliet’s grave. They fight, and Romeo kills Paris. He enters the tomb, sees Juliet’s inanimate body, drinks the poison, and dies by her side. Just then, Friar Lawrence enters and realizes that Romeo has killed Paris and himself. At the same time, Juliet awakes. Friar Lawrence hears the coming of the watch. When Juliet refuses to leave with him, he flees alone. Juliet sees her beloved Romeo and realizes he has killed himself with poison. She kisses his poisoned lips, and when that does not kill her, buries his dagger in her chest, falling dead upon his body.The watch arrives, followed closely by the Prince, the Capulets, and Montague. Montague declares that Lady Montague has died of grief over Romeo’s exile. Seeing their children’s bodies'

# get high freqency image
resultant_image = get_image_to_encode(image)

# flatten to 1D array
flat_image = resultant_image.flatten()
image_to_encode = image.flatten()

# get all indicies 
indices = np.argsort(flat_image)

# binarise message
string_binary = string_to_binary(string_to_encode)
encoded_image, original_pixel_values, indices = LSB_Matching_encode(image_to_encode, indices, string_binary)

decoded_message = LSB_Matching_decode(encoded_image, original_pixel_values, indices)

print(string_binary == decoded_message)


# Decode binary message
# need to fix
#def bin2text(s): 
#    return "".join([chr(int(s[i:i+8],2)) for i in xrange(0,len(s),8)])




# plot results
encoded_image = encoded_image.reshape((rows,cols,3))
fig, (ax_1, ax_2) = plt.subplots(1, 2, sharey=True, sharex=True)
ax_1.imshow(image)
ax_2.imshow(encoded_image)
