# Prompts: Kandinsky 6 Lite T2V / I2V

[Runner](README.md) · [JSON presets](prompt_presets.json). Smoke: 1 s, Baseline: 5 s. Таймлайн описательный; CFG=1 не использует negative guidance. Все звуковые описания внутри `<AUDCAP>` — пожелания генератору. Точную существующую запись добавляйте из своего аудиофайла при экспорте.

## orbit_woman — Woman / continuous camera orbit

### smoke

```text
Cinematic live-action fashion film, natural skin texture, restrained color grading, realistic cloth
physics. An adult woman, about 28 years old, stands in a circular stone courtyard. She wears a deep
emerald coat and a cream silk scarf. Archways and foreground columns establish clear depth. Warm
late-afternoon side light; stable exposure and soft moving highlights. The camera makes a small,
smooth 15-degree clockwise arc; keep her face centered. Do not attempt a full circle in one second.
Initial state: The woman faces the camera, both feet planted; hair tied loosely; scarf rests on her
right shoulder. One uninterrupted shot, about 1 seconds. From 0 to 0.35 seconds: Start from a
frontal three-quarter view. The camera begins its arc, a near column slides slowly across the
background. From 0.35 to 1 seconds: Continue the arc smoothly. She turns her eyes toward the lens
and the scarf lifts slightly, then begins to settle. Same face and outfit throughout. <AUDCAP>Quiet
courtyard air and a soft breeze. A brief silk rustle; no dialogue or music.<ENDAUDCAP>
```

### baseline

```text
Cinematic live-action fashion film, natural skin texture, restrained color grading, realistic cloth
physics. An adult woman, about 28 years old, stands in a circular stone courtyard. She wears a deep
emerald coat and a cream silk scarf. Archways and foreground columns establish clear depth. Warm
late-afternoon side light; stable exposure and soft moving highlights. The camera performs a smooth
approximately 160-degree clockwise orbit from a front three-quarter view to a rear three-quarter
view. Keep the woman centered at the same scale; archways pass with strong parallax. Initial state:
The woman faces the camera, both feet planted; hair tied loosely; scarf rests on her right shoulder.
One uninterrupted shot, about 5 seconds. From 0 to 1 seconds: Establish her face and emerald coat.
The camera starts moving; a foreground column passes along the edge. From 1 to 3.6 seconds: The
camera continues around her right side. She slowly turns her head to follow the lens, while a single
gust pulls her scarf outward. Her body remains planted. From 3.6 to 5 seconds: Arrive at the rear
three-quarter view. The scarf settles onto her shoulder and the camera eases to a stop; preserve her
silhouette and clothing. <AUDCAP>Soft courtyard ambience. One gentle gust, silk fluttering, distant
birds. The breeze fades; no speech or soundtrack.<ENDAUDCAP>
```

## strange_tram — Retrofuturist tram / practical creature effects

### smoke

```text
Surreal retrofuturist dark comedy, 1970s science-fiction cinema, tactile practical creature effects,
worn brass, painted miniatures, muted teal and amber, subtle film grain. A rain-wet tram stop in an
imaginary industrial city. An original pear-shaped, blue-skinned commuter with tiny spectacles and a
tan raincoat waits beside a brass ticket machine. He has two arms and two legs. Overcast daylight
mixed with warm tungsten light in the ticket machine. A short, slow push-in, without changing the
view angle. Initial state: The commuter holds one paper ticket. The ticket slot is shut. Steam
drifts behind the stop. One uninterrupted shot, about 1 seconds. From 0 to 0.4 seconds: The brass
ticket slot clicks open. The commuter freezes and raises one eyebrow. From 0.4 to 1 seconds: A small
mechanical tongue briefly taps the edge of his ticket, then retracts. The commuter blinks in deadpan
disbelief. <AUDCAP>One dry metallic click over quiet rain. A soft rubbery squeak and a tiny bell; no
speech.<ENDAUDCAP>
```

### baseline

```text
Surreal retrofuturist dark comedy, 1970s science-fiction cinema, tactile practical creature effects,
worn brass, painted miniatures, muted teal and amber, subtle film grain. A rain-wet tram stop in an
imaginary industrial city. An original pear-shaped, blue-skinned commuter with tiny spectacles and a
tan raincoat waits beside a brass ticket machine. He has two arms and two legs. Overcast daylight
mixed with warm tungsten light in the ticket machine. Track slowly left by about one metre as the
action unfolds; preserve the same subject and scale. Initial state: The commuter holds one paper
ticket. The ticket slot is shut. Steam drifts behind the stop. One uninterrupted shot, about 5
seconds. From 0 to 1 seconds: The commuter inserts the ticket halfway into the slot. From 1 to 2.8
seconds: A small mechanical tongue takes the ticket, curls it into a neat tube, and pushes it back
out. The creature watches without moving his feet. From 2.8 to 4.2 seconds: The commuter cautiously
takes the rolled ticket with the same hand and tilts his head. A tram headlamp slides into the far
background. From 4.2 to 5 seconds: Hold his puzzled expression as the slot closes with exaggerated
politeness. Maintain tactile material detail and a coherent street. <AUDCAP>Rain, faint tram rails
humming. A wet rubber squeak followed by delicate brass clicks. A distant tram approaches and rings
once. One final click; quiet rain, no narration.<ENDAUDCAP>
```

## mechanical_bloom — Biomechanical greenhouse / unfolding flower

### smoke

```text
Tactile miniature science-fiction film, stop-motion-inspired creature design with smooth motion,
copper mechanisms and translucent botanical tissue, detailed macro photography. An original small
flower made of copper ribs and translucent pale-blue petals grows from a cracked ceramic pot on a
workbench. The background is an unfocused greenhouse. Soft green backlight through glass with a warm
practical lamp on the workbench. A tiny sideways camera slide creating gentle foreground parallax.
Initial state: One closed flower bud; a drop of water hangs from a copper leaf. One uninterrupted
shot, about 1 seconds. From 0 to 0.5 seconds: The bud quivers and the first pair of petals
separates. From 0.5 to 1 seconds: The petals open a little farther and release one small puff of
glowing pollen. The pot and bench stay fixed. <AUDCAP>A quiet metallic tick. A soft pneumatic puff
and a delicate chime; no voice.<ENDAUDCAP>
```

### baseline

```text
Tactile miniature science-fiction film, stop-motion-inspired creature design with smooth motion,
copper mechanisms and translucent botanical tissue, detailed macro photography. An original small
flower made of copper ribs and translucent pale-blue petals grows from a cracked ceramic pot on a
workbench. The background is an unfocused greenhouse. Soft green backlight through glass with a warm
practical lamp on the workbench. A gentle 45-degree orbit around the flower at a fixed macro
distance. Initial state: One closed flower bud; a drop of water hangs from a copper leaf. One
uninterrupted shot, about 5 seconds. From 0 to 1 seconds: The closed bud slowly tilts toward the
warm lamp. From 1 to 3.4 seconds: Six translucent petals unfold in a clear sequence as the copper
ribs extend. One drop of water falls from the leaf. From 3.4 to 4.4 seconds: The flower opens fully,
releasing a single drifting cloud of luminous pollen. From 4.4 to 5 seconds: The flower settles and
the camera eases to a stop. Keep the same pot, leaf and bench. <AUDCAP>A faint mechanism starts
ticking. Delicate ratchet clicks, one water drop. A soft breath-like puff. The ticking subsides into
greenhouse ambience.<ENDAUDCAP>
```

## room_assembly — Scandinavian room / one bed assembly

### smoke

```text
Photorealistic Scandinavian furniture commercial, clean geometric composition, natural materials,
restrained palette. An empty sunlit room with white walls, a pale oak floor and one large window.
The room is unoccupied. Soft stable morning light from the left window. Completely locked-off wide
shot. Initial state: One unbranded flat-pack cardboard box in the center of the room. No other
furniture. One uninterrupted shot, about 1 seconds. From 0 to 0.3 seconds: The sealed cardboard box
begins to tremble gently. From 0.3 to 1 seconds: Its top flaps pop open and the first pair of wooden
bed rails rises a short distance above the box. The rest stays packed. <AUDCAP>A low brief cardboard
rattle. A soft pop and two wooden clacks. The room is otherwise quiet.<ENDAUDCAP>
```

### baseline

```text
Photorealistic Scandinavian furniture commercial, clean geometric composition, natural materials,
restrained palette. An empty sunlit room with white walls, a pale oak floor and one large window.
The room is unoccupied. Soft stable morning light from the left window. Completely static wide shot.
Keep the window, walls and floor identical. Initial state: One unbranded flat-pack cardboard box in
the center of the room. No other furniture. One uninterrupted shot, about 5 seconds. From 0 to 0.6
seconds: The box trembles and its top flaps open. From 0.6 to 2.5 seconds: Four wooden rails and a
few slats lift out, rotate into alignment and connect into one low minimalist bed frame. From 2.5 to
4 seconds: One mattress lowers onto the frame. A white duvet unrolls across it; a single mustard-
yellow throw lands at the foot. No other furniture appears. From 4 to 5 seconds: Hold the completed
bed in the otherwise uncluttered room. All pieces settle. <AUDCAP>Cardboard rustle and a small pop.
A short ordered sequence of wooden clicks. A cushioned thump and soft fabric rustling. A gentle warm
ambient chord, no speech.<ENDAUDCAP>
```

## mario_poster — Reference 1 / 3D poster character

Reference: `images-4.jpeg` (файл не включён).

### smoke

```text
Preserve the reference: polished 3D cartoon rendering, rounded proportions, bright red clothing,
crisp poster graphics. Use the supplied poster as the first frame and visual ground truth. Preserve
the orange left half, blue right half, radial rays, two small upper stickers and the exact character
design. Keep the original soft studio lighting and saturated orange-blue palette. Camera completely
static; the two upper stickers and radial background remain motionless. Initial state: The stocky
Mario-like cartoon character wears red overalls, a red cap with its existing emblem, white gloves
and brown shoes. Keep his identity and body shape. One uninterrupted shot, about 1 seconds. From 0
to 0.4 seconds: The main character blinks once and raises his eyebrows slightly. From 0.4 to 1
seconds: He makes a small cheerful nod; his cap and overall straps follow with a subtle secondary
bounce, then settle. Keep both hands and shoes in their original positions. <AUDCAP>A tiny friendly
cartoon exhale. A soft fabric creak and a light comic pluck. No spoken words.<ENDAUDCAP>
```

### baseline

```text
Preserve the reference: polished 3D cartoon rendering, rounded proportions, bright red clothing,
crisp poster graphics. Use the supplied poster as the first frame and visual ground truth. Preserve
the orange left half, blue right half, radial rays, two small upper stickers and the exact character
design. Keep the original soft studio lighting and saturated orange-blue palette. A very slow
approximately 3-percent push-in, no orbit; both upper stickers remain fixed graphic elements.
Initial state: The stocky Mario-like cartoon character wears red overalls, a red cap with its
existing emblem, white gloves and brown shoes. Keep his identity and body shape. One uninterrupted
shot, about 5 seconds. From 0 to 1 seconds: He notices the viewer and blinks, keeping the original
body shape and stance. From 1 to 2.7 seconds: He lifts his right gloved hand, on the left side of
the image, to the brim of his cap and tips it once. His other hand remains beside his body. From 2.7
to 4.2 seconds: He lowers that hand and gives a small proud shoulder bounce. The red overalls react
with subtle cloth motion; his shoes stay in place. From 4.2 to 5 seconds: Return to a relaxed near-
original pose. The background, stickers and existing emblems stay unchanged, with no new lettering.
<AUDCAP>A short curious cartoon hum. A gentle cloth rustle and one cheerful comic pluck. A quiet
breathy chuckle without intelligible speech. A light settling sound and then quiet.<ENDAUDCAP>
```

## mario_pipe_illustration — Reference 2 / drawn character on pipe

Reference: `3ad019e7defdecd45abf8be97e287972.jpg` (файл не включён).

### smoke

```text
Preserve the reference: warm paper background, hand-drawn ink outlines, muted colors, stippled
shading and limited 2D cartoon animation. Keep it illustrated throughout. Use the supplied square
drawing as the first frame. The heavy cartoon plumber sits on the same green pipe, holding the same
red-and-white mushroom and yellow flower. Keep the original flat warm illustration lighting and
textured cream paper. Locked camera; paper texture and floor cracks remain stationary. Initial
state: Preserve the red cap, blue overalls, white gloves, moustache, heavy body proportions,
mushroom, flower, pipe and floor cracks. One uninterrupted shot, about 1 seconds. From 0 to 0.4
seconds: The character blinks and looks toward the mushroom in his raised hand. From 0.4 to 1
seconds: The yellow flower nods gently once while his shoulders lift with a small silent chuckle.
Preserve the ink outline and original objects. <AUDCAP>A tiny amused nasal hum. A soft cloth creak
and a delicate spring sound; no speech.<ENDAUDCAP>
```

### baseline

```text
Preserve the reference: warm paper background, hand-drawn ink outlines, muted colors, stippled
shading and limited 2D cartoon animation. Keep it illustrated throughout. Use the supplied square
drawing as the first frame. The heavy cartoon plumber sits on the same green pipe, holding the same
red-and-white mushroom and yellow flower. Keep the original flat warm illustration lighting and
textured cream paper. Camera completely fixed; preserve the square paper illustration and keep all
props visible. Initial state: Preserve the red cap, blue overalls, white gloves, moustache, heavy
body proportions, mushroom, flower, pipe and floor cracks. One uninterrupted shot, about 5 seconds.
From 0 to 1 seconds: He glances from the mushroom to the yellow flower and lifts one eyebrow. From 1
to 2.7 seconds: His raised gloved hand tilts the mushroom toward his face without eating it; the
flower bends playfully toward the mushroom. Keep the props in their original hands. From 2.7 to 4.1
seconds: He leans back slightly and chuckles; his shoulders and belly move together with one
restrained squash-and-stretch bounce. The green pipe remains intact. From 4.1 to 5 seconds: He
settles toward the original seated pose and the flower makes one small final nod. Preserve the hand-
drawn style and paper texture. <AUDCAP>A low curious cartoon hum. A gentle springy boing and glove
rustle. A short warm comic chuckle; no words or music. A soft settling creak, then quiet.<ENDAUDCAP>
```

## mario_kandinsky_growth — Mario / Kandinsky mushroom / growth gag

Reference: `mario_kandinsky_mushroom_prepared.png` (файл не включён).

**Подготовка первого кадра (отдельный image editor):**

```text
Create a clean medium-wide first frame from the supplied Mario reference. Keep the same character
identity, red cap, overalls, white gloves and warm illustrated finish. He holds one large red
mushroom toward the viewer with the cap facing the camera. Place the exact supplied Kandinsky
wordmark on the mushroom cap as a composited decal; do not redraw or invent the lettering. Leave
generous empty space above his head and around his body for later growth. Keep the other hand empty
and the background simple.
```

### smoke

```text
Playful hand-drawn cartoon animation; preserve the supplied reference style, ink outlines, face,
moustache and costume colors. Use the prepared reference as the first frame. One cartoon plumber
holds one red mushroom whose cap already carries the supplied Kandinsky logo. Preserve the reference
appearance and background. Stable warm cartoon lighting, unchanged colors and exposure. Keep the
camera nearly fixed; a tiny backward drift compensates for the beginning of growth. Initial state:
He holds the mushroom close to his mouth, its existing Kandinsky decal facing the camera. His free
hand rests at his side. Plenty of headroom. One uninterrupted shot, about 1 seconds. From 0 to 0.4
seconds: The character takes one clear bite from the edge of the mushroom. Preserve the existing
logo on the remaining visible cap; the glove stays attached to the same hand. From 0.4 to 1 seconds:
A warm mint-colored pulse passes through him. His shoulders rise and his belly rounds slightly: only
the start of the power-up. His face and costume stay recognizable. <AUDCAP>One crisp cartoon crunch
on an energetic drum accent. A short rising power-up tone over an original syncopated new-wave funk
instrumental, dry drums, elastic bass and scratchy rhythm guitar; no vocals or quoted
melody.<ENDAUDCAP>
```

### baseline

```text
Playful hand-drawn cartoon animation; preserve the supplied reference style, ink outlines, face,
moustache and costume colors. Use the prepared reference as the first frame. One cartoon plumber
holds one red mushroom whose cap already carries the supplied Kandinsky logo. Preserve the reference
appearance and background. Stable warm cartoon lighting, unchanged colors and exposure. Begin in a
medium-wide view. After the bite, dolly smoothly backward and tilt upward gently so the growing
character remains fully inside the frame. No cuts. Initial state: He holds the mushroom close to his
mouth, its existing Kandinsky decal facing the camera. His free hand rests at his side. Plenty of
headroom. One uninterrupted shot, about 5 seconds. From 0 to 1 seconds: He looks at the viewer,
lifts the mushroom slightly and presents its existing Kandinsky decal toward the camera. Keep the
lettering already in the reference. From 1 to 1.8 seconds: He takes one bite. A small piece is
missing from the cap; his jaw closes naturally and he lowers the same hand. From 1.8 to 3.8 seconds:
One mint-colored pulse triggers a continuous cartoon power-up. He grows roughly one-and-a-half times
taller and becomes noticeably wider and rounder, with fuller cheeks, broader shoulders and a larger
belly. His overalls stretch with him; preserve his face, moustache, cap, two arms and two legs. The
camera pulls back steadily. From 3.8 to 5 seconds: Now oversized and comically plump, he plants his
feet, gives one satisfied belly bounce and grins. Ease the camera to a stop. Keep the final larger
body stable, with no extra objects or limbs. <AUDCAP>An original energetic early-1980s-style new-
wave funk groove: dry punchy drums, elastic bass, clipped rhythm guitar and short bright synth
accents; no vocals or familiar melody. A satisfying crunchy bite aligned to the beat. A rising
playful synth glissando and soft elastic fabric sounds over the steady funk groove. One low comic
landing thump, then a confident instrumental accent. No spoken dialogue.<ENDAUDCAP>
```

