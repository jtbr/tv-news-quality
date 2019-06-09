#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created Oct 17 2017
@author: Justin

Save snippets (text segments) from CC files to database
Snippets are the basis of the category classification tasks - to determine the
composition of news.

Currently:
Input: cc docs from mongodb
Output: snippets in mongodb

"""
## IMPORTS

import os, sys, re, math
from os import walk
import random
import pymongo
import time
from datetime import datetime,date,timedelta
from unidecode import unidecode
from nltk.util import ngrams as nltk_ngrams
from nltk.corpus import words
import nltk.data
sent_detector = nltk.data.load('tokenizers/punkt/english.pickle')


### PARAMETERS ###

#    from hashlib import md5
MONGODB_PORT = 25541

# take any BBC program broadcast on any channel (in practice, PBS), and create a
# separate station from them
SEPARATE_BBC = True
# take any Al Jazeera program broadcast on any channel (in practice, LINKTV), and
# add to ALJAZAM
SEPARATE_ALJZ = True # has the effect of extending series from 2012-03 to 2013-08
# (and overlapping 2013-08 to 2013-10)
# TODO: Could check 2013-08 to 2013-10 period overlap to see if they're significantly different
REMOVE_COMEDIES = True # removes comedy programs from the networks (not COM)
REMOVE_PARTY_PROGRAMMING = True # removes state of the union, debates, conventions, inauguration

# NOTE: DB contains 2009Nov-2010Dec from east coast stations, and 2011-present from west coast station

# search for programming from these networks
desired_stations = set(['CNN', 'CNBC', 'ALJAZAM', 'BLOOMBERG', 'MSNBC',
    'FOXNEWS', 'FBC', 'PBS', 'ABC', 'NBC', 'CBS', 'NPR', 'LINKTV', 'BBC'])
    # Note BBC optionally split out later from PBS
# final stations after splitting out
all_stations = ['ABC', 'NBC', 'CBS', 'PBS', 'CNN', 'FOXNEWS', 'MSNBC',
                'CNBC', 'FBC', 'BLOOMBERG', 'NPR', 'ALJAZAM', 'BBC']

english_words = set([word.lower() for word in words.words()])

### END PARAMETERS ###


# Regular expression definitions

#double checked all programs Oct 7 2016:
# local programs to omit:
localprog = re.compile('(?:ABC\s?7)|(?:Washington)|(?:wusa\s?9)|(?:9News\s?Now)|(?:Eyewitness)|' + \
    '(?:KQED)|(?:California)|(?:News\s?4)|(?:NBC\s?Bay)|(?:News\s?at\s?\d)|(?:Bay)|' + \
    '(?:Press\s?Here)|(?:KPIX\s?5)|(?:CBS\s?5)|(?:Mosaic)|(?:Up\s?to\s?the\s?Minute)|' + \
    '(?:To\s?the\s?Contrary)|(?:Assignment\s?7)|(?:Beyond\s?the\s?Headlines)|' + \
    '(?:Action\s?News)', flags=re.IGNORECASE) # (?: ...) is non-capturing parentheses

# coverage of events not produced by stations and potentially biased:
partyprog = re.compile('(?:Convention)|(?:Inauguration)|(?:State of the Union)|' + \
    '(?:(?<!Post[\s\-])(?<!Pre[\s\-])(?<!Great )(?<!Davos )Debate(?!\sAnalysis)(?!\sPreview))',
    flags=re.IGNORECASE) # NOTE: CNN has a regular program called State of the Union that should not be excluded

# late night comedy programs (not COM)
comedy = re.compile('(?:Jimmy Kimmel Live)|(?:The Tonight Show)|(?:The Late Show)|(?:Late Night)')


def hasNLines(N,filestr):
    """returns true if the filestr has at least N lines and N periods (~sentences)"""
    lines = 0
    periods = 0
    for line in filestr:
        lines = lines+1
        periods = periods + len(line.split('.'))-1
        if lines >= N and periods >= N:
            return True;
    return False;


def splitIntoSpeechParts(program_ccs):
    """Separate a program's CCs into 'snippet fragments'/'speech parts' for text analysis / classification. Returns a single 'speech' by a speaker if it's short enough, or segments of a 'speech' averaging around 100 characters if not, along with the line number it came from"""
    samples = []
#    print("\nSAMPLES\n")

    # process by lines (speeches by an individual speaker)
    for (line_no, line) in enumerate(program_ccs.splitlines()):
        #TODO: remove ads (?)
#        splitline = breakToAd.split(line)
#        if len(splitline)>1:
#            line = ''.join(splitline[0:-1]) # drop after the last upcoming ad marker

        line = line.replace('aa aa', '&#x266a; &#x266b;').replace('aa', '&#x266a;') # add back in music symbols
        sentences = sent_detector.tokenize(line)
        groupings = int(math.floor(len(sentences) / 3.)) + 1  # on average, want 3 sentences per sample
        speechLen = sum([len(sentence) for sentence in sentences])
        if (len(sentences)>1 and speechLen / groupings > 160):
            groupings += 1
        while (groupings > 1 and speechLen / groupings < 100): # want a minimum of 100 characters per group on avg
            groupings -= 1
        minNumSentencesPerSample = len(sentences) // groupings
        remainingSentences = len(sentences) % groupings
        first = True
        sample = ""
        i = 1
        # convert sentences into samples, combining as necessary
        for sentence in sentences:
            if first: #first sample on line (lines are formed from individual speeches)
                sample = "> " + sentence + ' '  # add a "new speaker" indicator to sample TODO: &gt; ?
                first = False
            else:
                sample += sentence + ' '
            if i < minNumSentencesPerSample:
                i += 1
                continue
            if i == minNumSentencesPerSample and len(sample) < 120 and remainingSentences > 0:
                # use one (and a maximum of one) of the extra sentences to pad out this sample
                i += 1
                remainingSentences -= 1
                continue
            samples.append((line_no, sample))
#            print(sample + "\n")
            i = 1
            sample = ""
        if sample:
            samples.append((line_no, sample))
            
    # second pass to consolidate very short samples
    samples2 = []
#    print("\n\nSAMPLES 2\n")
    samples_iter = iter(samples)
    nextSample = next(samples_iter, None)
    while nextSample:
        lineno = nextSample[0]
        sample = nextSample[1]
        nextSample = next(samples_iter, None)
        while nextSample and len(sample) < 80 and len(sample + nextSample[1]) < 100:
            sample += '  ' + nextSample[1]
            nextSample = next(samples_iter, None)
        samples2.append((lineno, sample))
#        print(sample + "\n")
        

    return samples2


# programs which are known to appear in the wrong station:
CBS_programs = {'60 Minutes','A Nation Remembers','Bay Area Focus With Susan Sikora',
    'CBS 5 Early Edition','CBS 5 Eyewitness News','CBS 5 Eyewitness News Early Edition',
    'CBS 5 Eyewitness News Special','CBS 5 Eyewitness News at 11','CBS 5 Eyewitness News at 11PM',
    'CBS 5 Eyewitness News at 530PM','CBS 5 Eyewitness News at 5AM','CBS 5 Eyewitness News at 5PM',
    'CBS 5 Eyewitness News at 630PM','CBS 5 Eyewitness News at 6AM','CBS 5 Eyewitness News at 6PM',
    'CBS 5 Eyewitness News at 730am','CBS 5 Eyewitness News at Noon','CBS 5 News',
    'CBS Evening News','CBS Evening News With Katie Couric','CBS Evening News With Russ Mitchell',
    'CBS Evening News With Scott Pelley','CBS Morning News','CBS News Campaign 2016',
    'CBS News Election Coverage','CBS News Sunday Morning','CBS News The First Presidential Debate',
    'CBS News The Second Presidential Debate','CBS News The Third Presidential Debate',
    'CBS News The Vice Presidential Debate','CBS Overnight News','CBS This Morning',
    'CBS This Morning Saturday','CBS Weekend News','CBS5 Eyewitness News  Pre-Game Show',
    'Campaign 2012 CBS News Coverage','Campaign 2016 CBS News Coverage of Election Night',
    'Campaign 2016 Democratic Convention','Campaign 2016 Democratic Debate',
    'Campaign 2016 Republican Convention','Campaign 2016 Republican Debate',
    'Democratic National Convention','Eyewitness News','Face the Nation','KPIX 5 News',
    'KPIX 5 News  Early Edition','KPIX 5 News Early Edition','KPIX 5 News Sat Morn Edition',
    'KPIX 5 News Saturday Morning Edition','KPIX 5 News Sun Morn Edition',
    'KPIX 5 News Sunday Morning Edition','KPIX 5 News and Pre-Game Show','KPIX 5 News and Pregame Show',
    'KPIX 5 News and the Fifth Quarter','KPIX 5 News at 11PM','KPIX 5 News at 11pm',
    'KPIX 5 News at 530PM','KPIX 5 News at 530pm','KPIX 5 News at 5AM','KPIX 5 News at 5PM',
    'KPIX 5 News at 5pm','KPIX 5 News at 600PM','KPIX 5 News at 630PM','KPIX 5 News at 630pm',
    'KPIX 5 News at 6AM','KPIX 5 News at 6pm','KPIX 5 News at Noon','KPIX 5 News on the CW',
    'KPIX 5 Noon News','Mosaic','Presidential Address','Presidential Debate',
    'Presidential Inauguration 2013','Religion  Ethics Newsweekly','Republican Debate',
    'Republican National Convention','State of the Union','State of the Union 2013',
    'State of the Union 2014','State of the Union 2015','State of the Union Address',
    'Sunday Morning','The Early Show','The Late Show With Stephen Colbert',
    'Through the Decades','To the Contrary With Bonnie Erbe','Up to the Minute',
    'Vice Presidential Debate'}

PBS_programs = {'2016 Presidential Debate','BBC World News','BBC World News America',
    'Broadcast News','Charlie Rose','Charlie Rose The Week','Democratic National Convention',
    'Frontline','Gubernatorial Debate','Inside Washington','KQED Newsroom','McLaughlin Group',
    'Moyers  Company','Nightly Business Report','PBS NewsHour','PBS NewsHour Convention Coverage',
    'PBS NewsHour Debates 2016 A Special Report','PBS NewsHour Weekend','PBS Newshour Election Night',
    'Pope Francis - The Sinner','Presidential Address','Presidential Debate',
    'Republican National Convention','State of the Union 2013','State of the Union 2014',
    'State of the Union Address','TED Talks','Tavis Smiley','The Contenders - 16 for 16',
    'This Week in Northern California','Vice Presidential Debate','Washington Week',
    'Washington Week With Gwen Ifill','Wilderness The Great Debate'}


def saveDocSnippets(doc, dbtable):
    """Obtain a list of snippets for a file, and save them to the provided dbtable"""
    station = doc['station']
    airdate = doc['airdate']
    airtime = doc['airtime']
    progname = doc['progname']
    cc_filename = doc['filename']
    docBody = doc['docBody'].encode('ascii','ignore')
    if not station in desired_stations:
        print "  Ignoring station: " + station + "  " + progname
        return

    print "  Processing: " + station + "  " + progname + "- " + airdate

    # ignore things that aren't where they should be
    #include WSJ on CNBC only (not carried nationally on any other network)
    #some PBS found in WUSA (CBS)
    #some CBS found in KNTV (NBC) (they appear to be NBC but may be non-news or local in some cases)
    #some found in KNTV (all sports or ads)
    if (station!='CNBC' and progname.find('Wall Street Journal')>=0) or \
       (station=='CBS' and progname in PBS_programs) or \
       (station=='NBC' and progname in CBS_programs) or \
       (station=='NBC' and (progname.find('BBC')>=0 or progname.find('Frontline')>=0 or progname.find('McLaughlin Group')>=0)):
        print "   - Skipping."
        return # skip/ignore

    # ignore optional categories as requested
    if REMOVE_COMEDIES and comedy.findall(progname):
        print "   - Skipping comedy."
        return # ignore it
    if REMOVE_PARTY_PROGRAMMING and partyprog.findall(progname):
        if not (station=='CNN' and progname == 'State of the Union'): #regular program
            print "   - Skipping non-station (party) programming"
            return # ignore it

    # special cases:
    if SEPARATE_BBC and progname.find('BBC')>=0:
        print "   - Allocating program to BBC station."
        station = 'BBC'
    if SEPARATE_ALJZ and progname.find('Al Jazeera')>=0:
        print "   - Allocating program to Al Jazeera station."
        station = 'ALJAZAM'

    if not station in all_stations:
        # remove LINKTV in practice
        print "   - Ignoring unseparated program on station: " + station + "  " + progname
        return

    # ignore programs that don't appear to be in english (probably gibberish)
    wc = 0
    eng_wc = 0
    for word in docBody.split(" "):
        if word.lower() in english_words: # will miss ends of sentences, possessives etc
            eng_wc += 1
        wc += 1
        if wc >= 200:
            break
    if (eng_wc / float(wc)) < 0.5:
        print "   - Less then 50% english words! Skipping"
        return            
        

    spTuples = splitIntoSpeechParts(docBody) # speechPartTuples (line #, speech part)

    count = 0
    nSpeechParts = len(spTuples)
    if nSpeechParts < 5:
        print "   - Fewer than 5 speech parts in document: %s. How?" % cc_filename
    for idx in range(0,nSpeechParts, 3):
        if idx + 2 >= nSpeechParts:
            # skip the last 1-2 lines; not enough for a snippet.
            break

        priorspeechpart  = spTuples[idx][1].strip().replace('"', "'")
        speechpart       = spTuples[idx + 1][1].strip().replace('"', "'")
        nextspeechpart   = spTuples[idx + 2][1].strip().replace('"', "'")

#       DON'T skip; we're treating the snippets as a group of 3 for categorization.
#        if len(speechpart) < 50:
#            # skip very short snippets
#            continue

        startLine = spTuples[idx][0] # unknown

        airdatetime = datetime.strptime(airdate+'T'+airtime, '%Y%m%dT%H%M%S')

        # samplerecord = cc_filename + ',' + str(count) + ',' \
        #                     + str(startLine) + ',' + station + ',' \
        #                     + str(airdatetime) + ',"' + priorspeechpart + '","' \
        #                     + speechpart + '","' + nextspeechpart + '"'
        # print samplerecord

        dbtable.insert_one({'filename': cc_filename, 'station': station,
                            'file_idx': count, 'file_lineno': startLine,
                            'airdatetime': airdatetime, 'snippet_part1': priorspeechpart,
                            'snippet_part2': speechpart, 'snippet_part3': nextspeechpart})
        count += 1



def main(args):
    if len(args)>1:
        print "USAGE: docs2snippets"
        return

    # TODO: use a real seed in production!
    random.seed(a=3) # seed randomizer for predictability

    with pymongo.MongoClient('localhost', MONGODB_PORT) as mclient:
        mdb = mclient.ccdb
        mdocs = mdb.docs
        mdocs.create_index('filename', unique=True) #creates index if not already there
        mdocs.create_index('airdate')
        mdb2 = mclient.snippetdb
        msnippets = mdb2.snippets
        msnippets.create_index([('filename', pymongo.ASCENDING), ('file_idx', pymongo.ASCENDING)], unique=True)
        msnippets.create_index([('station', pymongo.ASCENDING)])

        #note, requires mondo 3.2+
        #for doc in mdocs.aggregate({"$sample": {"size": RANDOM_FILES}}):

    #   here is a random sampling approach using random selection with replacement:
    #   Note: for large sample/collection sizes this is somewhat impractical since each doc has to skip() a large number of documents to be returned
    #    for i in range(RANDOM_FILES):  # get RANDOM_FILES #/docs, with replacement
    #        doc = mdocs.find()[random.randrange(count)]

        start_time = time.time()
        count = mdocs.count()
        print "Generating snippets for all eligible programs out of %d" % count

        all_docs_cursor = mdocs.find()
        index = 0
        for record in all_docs_cursor:
            saveDocSnippets(record, msnippets)
            index += 1
            if index % 100 == 0:
                elapsed_time = time.time() - start_time
                time_left = (elapsed_time / float(index)) * (count - index)
                print "\nOn record %d of %d;  estimated %s remaining\n" % (index, count, str(timedelta(seconds=time_left)))

        print "Done!"

if __name__ == "__main__":
    main(sys.argv)
