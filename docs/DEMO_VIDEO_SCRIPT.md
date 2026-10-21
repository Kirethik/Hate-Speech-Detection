# Civitas AI — Demo Video Script (under 6 minutes)

| # | Page | Time | Running total |
|---|---|---|---|
| 0 | Opening | 0:20 | 0:20 |
| 1 | Overview | 0:30 | 0:50 |
| 2 | Analyzer | 1:30 | 2:20 |
| 3 | Rewrite | 0:30 | 2:50 |
| 4 | Voice Studio | 1:20 | 4:10 |
| 5 | Live Room | 0:45 | 4:55 |
| 6 | Moderation + Analytics | 0:25 | 5:20 |
| 7 | Evaluation + Settings | 0:20 | 5:40 |
| 8 | Closing | 0:15 | 5:55 |

🎙️ **SAY** = voice-over · 🖱️ **DO** = on-screen action · 💬 **INPUT** = what to type, play or speak

**Speed tips:** cut out every loading wait while editing; you narrate over the result, not the spinner. Paste text instead of typing it. Keep each 🎙️ block to the time in the table.

> Put this card on screen at the start: *"This video contains examples of hate speech, shown only to demonstrate detection."*

---

## Before you record

1. Run `scripts\run_dev.ps1`, start Ollama, and check that the top-right pill says **Models ready**.
2. Run every input below once as a warm-up. The first run loads the models and is slow, and a test run confirms each example gets flagged.
3. Run `.venv\Scripts\python.exe scripts\evaluate.py --split val --per-language 300 --with-llm` so the Evaluation page has numbers.
4. Record the Voice Studio clip (Section 4) in advance in a quiet room.
5. For the Live Room you need a second person or device, and both of you must wear headphones.

---

## 0. Opening (0:20)

🎙️ **SAY**
> "Online conversations happen in many languages, and often a mix of them. Most moderation tools understand only English text, and when they find something harmful, all they can do is delete it. Civitas AI detects hate speech in text and speech across six Indian and South Asian languages, explains why it was flagged, and suggests a better way to say it, even in the speaker's own voice. Everything runs locally."

---

## 1. Overview (0:30)

🖱️ **DO:** Click **Overview**. Move the cursor across the three cards.

🎙️ **SAY**
> "Civitas does three things. **Detect:** fast multilingual classifiers screen every sentence, and a local language model judges the borderline ones in context, names the group being targeted, and highlights the exact words responsible. **Rewrite:** it suggests a cleaner version and a calm counter-reply in five tones. Every suggestion is safety-checked before it is shown. **Speech:** it transcribes audio, replaces only the hateful sentences in the speaker's own voice, and stitches the audio back together seamlessly."

🖱️ **DO:** Click **Open the Live Analyzer →**.

---

## 2. Analyzer (1:30)

### 2a. English hate speech (0:40)

🖱️ **DO:** Click **Type Text**, paste the input, click **Analyze**. Then click **Reveal Content**.

💬 **INPUT**
```
People of that religion are vermin and they should all be thrown out of this country.
```

🎙️ **SAY**
> "The verdict is **Hate Speech**, the target is **Religion**, and the *Why* section explains the decision in plain language. Hateful text is blurred by default to protect moderators. When I reveal it, the words that drove the decision are highlighted. Below are two alternatives: a rewrite for the speaker, and a counter-narrative a moderator could post. Both are ready to copy."

### 2b. Nuance: offensive is not the same as hate (0:15)

💬 **INPUT**
```
This referee is a complete idiot, what a damn joke of a match.
```

🎙️ **SAY**
> "This one is rude, but it attacks a person's decision, not a group. Civitas labels it **Offensive**, not hate, so moderators can respond proportionately."

### 2c. Multilingual (0:25)

Run Hindi, then the code-mixed line.

💬 **INPUT — Hindi**
```
इस धर्म के लोग कीड़े हैं, इन्हें इस देश से बाहर निकाल देना चाहिए।
```
*(Meaning: "People of this religion are insects, they should be thrown out of this country.")*

💬 **INPUT — Hinglish (code-mixed, Roman script)**
```
Ye migrants log bilkul gande hain yaar, inko country se bahar nikalo.
```
*(Meaning: "These migrants are totally filthy, throw them out of the country.")*

🎙️ **SAY**
> "In Hindi, it's flagged and the alternatives come back in Hindi, in the same script. And this code-mixed message, Hindi and English in Roman script, is exactly where most tools fail. Civitas catches it too. The same works in Urdu, Tamil, Telugu and Malayalam."

*Backup inputs if Hindi or Hinglish doesn't get flagged in your warm-up:*
- Tamil: `இந்த மதத்தைச் சேர்ந்தவர்கள் அழுக்கானவர்கள், இவர்களை நாட்டை விட்டு வெளியேற்ற வேண்டும்.`
- Roman Urdu: `Is mazhab ke log jaahil aur ganday hain, inhein mulk se nikaal dena chahiye.`
- Malayalam: `ഈ മതക്കാർ വൃത്തികെട്ടവരാണ്, ഇവരെ നാട്ടിൽ നിന്ന് പുറത്താക്കണം.`

### 2d. Live mic (0:10)

🖱️ **DO:** Click **Live Mic** → **Start Recording**, speak the line below, then click **Stop & Analyze**.

💬 **SPEAK:** *"Honestly, those migrants are parasites and they should all be sent back."*

🎙️ **SAY**
> "The same analysis works on live speech, or on an uploaded audio file."

---

## 3. Rewrite (0:30)

🖱️ **DO:** Click **Rewrite**, paste the input, choose **Calm**, and click **Generate**. Then switch to **Firm** and generate again.

💬 **INPUT**
```
These migrants are ruining our city, they are all criminals and should be kicked out.
```

🎙️ **SAY**
> "The Rewrite Studio helps people say it better. I choose a tone: calm, factual, firm, empathetic or professional. The calm version keeps the real concern, the state of the city, and drops the claim that a whole group are criminals. The **safety check passed** badge confirms the rewrite was re-scored. The text is editable, so a person always has the final say. *Firm* gives a more direct version for the same message."

---

## 4. Voice Studio (1:20)

**Clip to record in advance** (~20 seconds, with a short pause between sentences):
> "Good evening everyone, and thank you for coming to tonight's community meeting.
> We need to talk about the new parking rules near the market.
> Honestly, those migrants are parasites and they should all be thrown out of this town.
> Anyway, the council will vote on the proposal next Friday."

🖱️ **DO:** Click **Voice Studio** → **Upload file** → drop the clip. Leave **Rewrite** selected and tick the **consent** box.

🎙️ **SAY**
> "Voice Studio cleans recordings. Here is a community meeting with one hateful sentence. I choose a rewrite and confirm consent. Civitas never re-voices anyone without it."

🖱️ **DO:** Click **Clean this audio**. Cut the wait in editing, leaving about 2 seconds of the stage tracker.

🎙️ **SAY**
> "It transcribes, scans each sentence, rewrites the hateful one, learns the voice, and stitches it back together."

🖱️ **DO:** Play the **Original** waveform over the marked region, then the **Cleaned** waveform over the same region. Then scroll to the transcript.

🎙️ **SAY**
> "Here's the original… and here's the cleaned version. It's the same speaker and the same meeting, but the hateful sentence is replaced with a constructive one in their own voice. Every edit is marked on the waveform. The transcript shows what was removed, what replaced it, the target group and the reason. I can download the cleaned audio and a full edit report for auditing. Speakers can also enroll their voice once for even better quality."

---

## 5. Live Room (0:45)

**Setup:** you are on the main screen and a second person joins on another device. Both of you wear headphones.

🖱️ **DO:** Click **Live Room**, enter your name, tick consent, click **Join room**, then **Copy invite link**. The second person joins with the link.

💬 **CONVERSATION** (one sentence at a time, with a short pause after each):
> **You:** "Hi Ravi, did you see the news about the new housing project?"
> **Ravi:** "Yes, what do you think?"
> **You:** "Honestly, those migrants are parasites and they should all be sent back."
> **Ravi:** "What should the council do about it?"

🎙️ **SAY**
> "The Live Room does this in real time. Ravi didn't hear what I actually said. He heard a rewritten version, in my voice, a few seconds later. The caption is tagged **Re-voiced**, and I see what others heard instead, so moderation is a visible nudge, not silent censorship."

---

## 6. Moderation + Analytics (0:25)

🖱️ **DO:** Click **Moderation**, then click **✓ Approve** on one row and **Override** on another. Then click **Analytics**.

🎙️ **SAY**
> "Humans stay in control. In the Moderation Center, reviewers approve the AI's decisions or override false positives, and every action is saved as feedback. Analytics shows the volume analysed, the detection rate, moderator reviews, and a breakdown by language."

---

## 7. Evaluation + Settings (0:20)

🖱️ **DO:** Click **Evaluation** and point at the metrics and the per-language table. Then click **Settings**.

🎙️ **SAY**
> "Accuracy isn't taken on trust. These numbers come from a real run on labelled data: recall, false positive rate, results per language, and how many rewrites passed the safety gate. And Settings confirms everything, from the classifiers and the language model to speech and voice, is running locally on this laptop."

*Tip: read one real number from your screen, for example "…recall of X percent".*

---

## 8. Closing (0:15)

🎙️ **SAY**
> "Civitas AI detects hate in text and speech across six languages, explains every decision, and offers a better way to say it, in the speaker's own voice. It isn't about silencing people. It's about helping them say what they mean, without the hate. Thank you for watching."

---

## Don't say on camera (not in the current build)

- That the AI audio is **watermarked**. There is no watermarking code.
- That an **NLI / lexicon** second stage ran. The pipeline is classifiers plus the LLM judge.
- Don't open **Model Diagnostics → Severity Probabilities**. It's empty in this build.
