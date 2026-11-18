# Demo video script (3–5 minutes)

Record the deployed Render URL in a laptop browser, plus 20 seconds on a phone. Open `/api/health` a minute before recording so the free service is awake.

| Time | Screen | What to say |
|---|---|---|
| 0:00–0:25 | Operator page, empty | "ParkSense counts free parking bays from one fixed camera. The operator outlines the bays once; after that every image is checked in under a second. It runs free on Render, and images are never stored." |
| 0:25–1:15 | **Case 1: UFPR05 · cloudy (unseen car park).** Choose the sample, show the outlines, click *Check occupancy*. | "This car park was never used for training. 46 bays: 26 occupied, which matches the hand count exactly, with one bay marked not sure. Every bay carries a letter as well as a colour, so it works for colour-blind users. Server time is under 100 ms." |
| 1:15–1:45 | Click *Outline bays*, draw one bay with four clicks, *Save layout*. | "Setting up a new camera: four clicks per bay, then save the layout and reuse it for every frame." |
| 1:45–2:25 | **Case 2: occupancy over time.** Click *Run the sample day (UFPR05)*. | "Sixteen frames across a Monday. The red line is the model and the dashed line is the hand-labelled truth. They overlap from the empty early morning through the full middle of the day." |
| 2:25–3:25 | **Case 3, the failure: CNRPark-EXT camera 5 · sunny, low camera.** Run with *CNN*, then *CNN + baseline*. Zoom in on the far rows. | "A different country and a low camera that looks across the car park rather than down on it. The truth is 43 occupied bays; the CNN finds 28, plus 4 it is not sure about. It misses cars it sees from behind, because it learned cars from above, and most of those mistakes are confident, so 'not sure' doesn't catch them. CNN + baseline sends 11 bays to 'not sure', so more reach the operator. In our tests, correcting three frames from a new camera raises accuracy from 82% to 91%. This is why a new camera needs a check before its counts go public." Click two wrong bays to override them; the counts update. |
| 3:25–3:50 | *Publish counts* → Public view, then the same page on a phone. | "Only the numbers are published: no image, no plates, no people." |
| 3:50–4:30 | Report figures: arms chart, stress table. | "Results: 99.5% at the training car park, 96% on unseen PKLot car parks against 90% for the classical baseline, 81% on a new camera set, 91% after three corrected frames. The CycleGAN rain translator kept every label but didn't improve accuracy, because rain barely changes a bay at this camera. Simulated fog and night push the model to chance, so the app now warns on dark, foggy or blurred frames." |
| 4:30–4:45 | Help page / model card | "Limits and intended use are on the help page and in the model card." |
