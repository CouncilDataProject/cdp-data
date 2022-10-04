# CDP SIG Capstone
Plan for Victoria

## 1:1 and Lab Meetings

I am pretty flexible to meet anytime. I would like to be consistent
in our meeting schedule though. I understand missing and moving around
a meeting here or though though.

### 1:1 Meetings

If you prefer in-person, I would prefer Tuesday morning for 30 minutes
anytime between 9:30am and 11:30am.

If you prefer remote, I would prefer Monday morning for 30 minutes
anytime between 9:00am and 12:00pm.

### Lab Meetings

PIT lab meeting is Friday from 3:00pm - 4:00pm each week. If you can make that,
it would be awesome if you could come give a brief update each week and then
join the conversation as others give updates as well!

## Schedule of Work

### Week 1 -- Get Set Up

1. Install conda, install Just
2. Fork cdp-data
3. Create conda environment
4. Install cdp-data locally
5. Create a Jupyter notebook with some random cdp-data function calls

### Week 2 -- Exploratory Analysis

1. Using cdp-data and your own code
2. Try to answer the following questions:
    1. blah blah
    2. blah blah

### Week 3 -- Add your Analysis Functions into CDP-Data

1. Generalize the code you wrote in the notebook to work for any city
2. Make a pull request with your new additions
3. Make sure that the code is "linted", "formatted", and "type-checked"

### Week 4 -- An Intro to Special Interest Groups

1. Skim through transcripts of council meetings on councildataproject.org for
   multiple municipalities looking for references to people and organizations
   which influenced legislation (bills, resolutions, etc.).

   This may be stated as:
   1. "I want to thank X, Y, and Z for their work no craft this legislation..."
   2. "We heard from A and B while working on this legislation..."

   Record these references somewhere so we can look them up later.

   1. Event ID
   2. Session ID
   3. Sentence Start Timestamp
   4. Text

### Week 5 -- NLP and NERs

1. Use cdp-data to get a dataset of Seattle City Council meetings (~last six months).
2. Using spacy, use named entity recognition (NER) on each sentence of each transcript
   and create a table of:
   
   1. Event ID
   2. Event Datetime
   3. Body Name
   4. Session ID
   5. Sentence Index
   6. Sentence Text
   7. NER Label
   8. NER Text

   So that if there are multiple tagged entities in the same sentence, they each
   are given their own row.

### Week 6 -- Visualization

1. Using the data from week 5, create a couple of visualizations in a notebook
   breaking down which people and organizations are commonly referenced in meetings.

   1. How does this breakdown by body? (do different bodies have different orgs)
   2. How does this breakdown by NER label? (are there more references to people
      or organizations)
   3. Add one of your own!
