#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created Feb 2019
@author: Justin

Quickly vets a csv file of mturk label responses to uncover blatantly 
bad responses for rejection
"""
import os, sys
import numpy as np
import pandas as pd
import textwrap as tw
from builtins import input
from common import Label_DbFields
from datetime import tzinfo, timedelta, datetime
from email.utils import parsedate_tz, mktime_tz # python 3.3+ easier to use parsedate_to_datetime

# Only vet labels accepted after this date (Year-month-day hour-minute-second).
# Useful if you have to do multiple stages of vetting because of many rejections the first round.
VET_ACCEPT_AFTER = datetime(2019, 2, 6, 1, 0, 0)

Answer_Columns = ['Answer_' + label[6:] for label in Label_DbFields]
Suspect_Workers = ['AM4YL7TDQ1P4I', 'AVNW96ITB1X0W', 'A2RSOGTSR83XKO', 'A1TPCNI0K1EVYD', 'A9SQ8RTZZLVXQ', 'A1A26520MMTUSF', 'A8KM81W0MZT2G', 'A1VOKVBD97VXVC']

def getDateTime(dtstr):
    ''' returns a naive datetime UTC-time object from the given date string in the format:
        'Tue Feb 05 13:24:52 PST 2019'
        
        Note: datetime.strptime(tstr, "%a %b %d %H:%M:%S %Z %Y") won't work as only 
        'GMT' and 'UTC' are supported due to a longstanding python bug.

        In python 3, use this to return an aware datetime object:
        email.utils.parsedate_to_datetime(dtstr)
    '''
    return datetime.fromtimestamp(mktime_tz(parsedate_tz(dtstr)))
    

def showVetRec(rec):
    print("\n\n%s  %s  %d  (%ds)\n" % (rec.WorkerId, rec.AssignmentId, rec.Input_idx, rec.WorkTimeInSeconds))
    
    print("  %s\n\n  %s\n\n  %s\n" % 
        ('\n'.join(tw.wrap(rec.Input_priorsample,50)), 
         '\n'.join(tw.wrap(rec.Input_sample,50)), 
         '\n'.join(tw.wrap(rec.Input_nextsample,50))))
    
    for col in Answer_Columns:
        print('%15s: %s' % (col[7:], getattr(rec, col)))
    print(getattr(rec, 'Answer_comment'))

    response = input('\n[ok]? reject comment? (mod)ify later? stop? ')
    if response == '':
        response = 'ok'

    return (rec.HITId, response)


def addAcceptReject(data, responses):
    for HITId, assessment in responses:
        if assessment == 'ok':
            data.loc[data['HITId'] == HITId,'Approve'] = 'x'
        else:
            data.loc[data['HITId'] == HITId,'Reject'] = assessment


def vetLabels(csvFile):
    data = pd.read_csv(csvFile)
    data.columns = data.columns.str.replace('.', '_') # . has special meaning in pandas. replace with _.
    #data = data.astype({'Approve': str, 'Reject': str}) # creates string 'nan's
    data = data.fillna('').astype({'Approve': str, 'Reject': str})

    print("Showing SUSPICIOUS WORKERS")
    aborting = False
    responses = []
    modify_later = []
    accepted_without_mods = []
    for row in data.itertuples():
        if (getDateTime(row.AcceptTime) < VET_ACCEPT_AFTER):
            continue
        if (row.WorkerId in Suspect_Workers) and not row.Approve and not row.Reject:
            response = showVetRec(row)
            if response[1] == 'stop':
                aborting = True
                break
            elif response[1][:3] == 'mod':
                modify_later.append(row.Input_idx)
                responses.append((response[0], 'ok'))
            else:
                responses.append(response)
                if response[1] == 'ok':
                    accepted_without_mods.append(row.Input_idx)

    addAcceptReject(data, responses)

    responses = []
    if not aborting:
        print("Showing rapid responses (<25s)")
        data.sort_values(by=['WorkTimeInSeconds'], ascending=True, inplace=True)
        for row in data.itertuples():
            if row.WorkTimeInSeconds >= 25:
                break
            if (getDateTime(row.AcceptTime) < VET_ACCEPT_AFTER):
                continue
            if not row.Approve and not row.Reject:
                response = showVetRec(row)
                if response[1] == 'stop':
                    aborting = True
                    break
                elif response[1][:3] == 'mod':
                    modify_later.append(row.Input_idx)
                    responses.append((response[0], 'ok'))
                else:
                    responses.append(response)
                    if response[1] == 'ok':
                        accepted_without_mods.append(row.Input_idx)

        addAcceptReject(data, responses)
        #responses = []

    print('\nidx numbers to revisit for tiebreakers in this batch:')
    print(modify_later)

    print('\nidx numbers accepted without modification in this batch:')
    print(accepted_without_mods)
    
    data.sort_values(by=['AcceptTime'], ascending=True, inplace=True)
    return data


def main(args):
    if len(args) < 2:
        print("vetLabels [mturk_label_csv]")
        sys.exit(1)
    
    mturk_csv = args[1]
    output_csv = mturk_csv[:-4] + '-acceptreject.csv'
    if not os.path.exists(mturk_csv):
        print("csv not found")
        sys.exit(2)
    
    dataframe = vetLabels(mturk_csv)

    print("\nWriting output csv " + output_csv)
    dataframe.to_csv(output_csv, index=False)


if __name__ == "__main__":
    main(sys.argv)
