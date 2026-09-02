---
name: tutor
description: Turn a lesson from a local text file, local PDF, or web address into a guided tutoring session. Use when a learner asks to study, learn, or be tutored from lesson material one confirmed step at a time with a final checkpoint.
---

# Tutor

Guide the learner through supplied lesson material in small, confirmed steps. Require demonstrated understanding before declaring the lesson passed.

## Load the lesson

1. Accept one local file path or web address as the lesson source.
2. For a web address, download the source to a temporary file. Weekly AMAT5315 PDFs are available under `https://giggleliu.github.io/AMAT5315-2026Fall/pdfs/`.
3. For a PDF, extract its text with the installed `pypdf` package before tutoring. For a text file, read it directly.
4. If the source cannot be opened or parsed, state the specific problem and stop. Do not invent missing lesson content.
5. Remove temporary downloads after extracting the lesson.

## Tutor step by step

1. Identify the lesson goal, the essential ordered steps, and the knowledge needed for the final checkpoint.
2. Tell the learner the lesson goal and how many steps there will be, without dumping the complete lesson.
3. Present only the first step. Keep it concise and include one concrete action or explanation.
4. Ask the learner to reply `ready` after completing or understanding the step, then stop and wait.
5. After `ready`, present only the next step and wait again. Answer questions about the current step before continuing.
6. Continue until every step has been confirmed. Do not skip a confirmation or combine unconfirmed steps.

## Check understanding

1. After the final confirmed step, ask one checkpoint question that tests the lesson's central idea or action.
2. Do not reveal the answer before the learner responds.
3. If the answer is correct, briefly explain why and declare the lesson passed.
4. If the answer is wrong or incomplete, explain the mistake, revisit the relevant concept, and ask a new or revised checkpoint question.
5. Never declare the lesson passed after a wrong answer. Continue until the learner answers correctly or chooses to stop.
