/* MTurk common news JS */
// Category, description, label
var category_tuples = [
    ['Transitions, greetings, farewells', "Beginnings and ends of segments, transitions between reports, 'teasers' for upcoming stories, chit-chat, etc. (nothing of substance)", 'transitions'], // other/unknown
    ['Elections / Campaigns - substantive', "Candidates' and parties' plans/platforms/policies or expert opinions about policies, candidates' records/background, interviews with or speeches by candidates, election results (U.S.)", 'elections_hard'], // hard  (/meaningful/educational)
    ['Elections / Campaigns - fluff', "The horse race (who's leading), polls/standings, speculation about outcomes, campaign tactics/strategy, talking heads/commentators (about an election), campaign updates (U.S.)", 'elections_soft'], // soft
    ['Business, Finance and Economics', "News about the economy, markets, and business strategy, performance or business climate", 'business_economics'], // hard
    ['Science, Technology &amp; the Environment', "Scientific &amp; technological reporting; nature &amp; environmental coverage (incl. climate, but not weather; incl. medical sciences but not health care)", 'science_tech'], // hard
    ['Government affairs and Politics', "US or International Government actions, functioning, policies, and impacts (including military, except use of force), and coverage of (non-U.S.-election-related) Politics", 'government'], // hard
    ['Entertainment / Arts &amp; Celebrity news', "News relating to celebrities and the music / arts / literature / entertainment industries", 'entertainment'], // soft
    ['Sports', "News about Sports and Games", 'sports'], // soft
    ['Weather / Traffic', 'Current reports or forecasts of traffic or weather (but not the government response to weather, climate science, transport policy, or impacts of extreme weather events)', 'weather'], // soft
    ['Consumer affairs, Products &amp; Services', 'News reports about new or notable products or services, or reviews', 'products'], // soft
    ['Anecdotal and Human Interest', 'Stories which are perhaps interesting but mostly inconsequential. Usually quick/easy to produce. Often with only local significance and unknown or negligible wider impacts.', 'anecdotes'], // soft
    ['Current Events', 'Notable or consequential recent (at the time of presentation) news events. Terrorism, crime, protests, significant achievements, war/military use of force, etc.', 'current_events'], // hard
    ['Social Issues and Cultural Coverage', "Societal or cultural themes; concerning communities, cities, health, society, religion, gender, race, human interactions, or trends (which don't fit into earlier categories)", 'cultural'], // hard?
    ['Ads/sponsorships, self-promotion', 'Contents of apparently paid ads, sponsors that fund the network, or network self-promotion (such as for other programs)', 'ads'], // other/unknown
    ['None of the above/Impossible to classify', "Content that doesn't fit in any of these categories (please explain in comment); or incomprehensible, non-English, jibberish, or missing text; impossible to determine", 'none'], // other/unknown
];

// id, name, autosized bool, [[option, option name, option text], ..]
var supplemental_questions = [
    ['UsForeignInputs', 'usforeign', false, [['domestic', 'Domestic', "US domestic coverage"],['foreign', 'Foreign', "Non-US, international/global coverage, or US gov't foreign relations"], ['unknown', 'Unclear', "Unknown/unclear/neither/both"]]],
    ['FactOpinionInputs', 'factopinion', false, [['fact', 'Fact-based', "Fact or evidence-based statements (regardless of your view of them, or whether the statements are correct)"], ['opinion', 'Opinion', "Editorial, opinion, conjecture, speculation, possibilities"], ['other', 'Other', "Other/neither/unknown (e.g. interviews)"]]],
    ['InvestigativeInputs', 'investigative', true, [['investigative', 'Investigative or in-depth reporting', "Hard-hitting or in-depth factual reports and investigations that take time and effort to produce (e.g. providing history/background/context, exposing corruption or scandals); serious policy analysis"], ['noninvestigative', 'Other', "Anything else (so most stories); non-investigative reporting, including reports about others' investigations"]]],
    ['ToneInputs', 'tone', false, [['positive', 'Positive tone', "Positive in tone or sentiment (regardless of your view of the issue)"], ['neutral', 'Neutral', "Neutral in tone or sentiment"], ['negative', 'Negative tone', "Negative in tone or sentiment (regardless of your view of the issue)"]]],
    ['EmotionInputs', 'emotion', true, [['scary', 'Scary or outrageous', "Appeals to emotion: Stories which are scary, shocking, or outrageous in their topic or presentation (whether or not you feel that way)"], ['pleasant', 'Feel-good', "Appeals to emotion: Pleasant, sunny, inspirational or feel-good stories"], ['neither', 'Neither', "Neither scary/outrageous nor feel-good, or no appeals to emotion (Most stories are neither)"]]]
    // live coverage of unfolding events? (no, hard to tell)
];

function activateQuestion(question_id) {
    $(question_id).show();
    $(question_id + " input").attr('required', '');
}
function deactivateQuestion(question_id) {
    // Hide questions, make them not required, deselect all options and show them as deselected
    $(question_id).hide().children().removeClass('active');
    $(question_id + " input").removeAttr('required').prop('checked', false);
}
$(function() {
    // Populate categories
    var instr_table_entries = '',
        button_labels = '';
    for (i=0; i<category_tuples.length; i++) {
        var category = category_tuples[i][0],
            desc = category_tuples[i][1],
            label = category_tuples[i][2];
        entry = '<tr><td>'+category+'</td><td>'+desc+'</td></tr>';
        instr_table_entries += entry

        label = '<label class="btn btn-xs btn-default" data-toggle="tooltip" ' +
            'title="'+desc+'"> <input id="'+label+'" name="category" required="required" ' +
            'type="radio" value="'+label+'" />'+category+'</label>';
        button_labels += label
    }
    $('#CategoryTable').append(instr_table_entries);
    $('#CategoryInputs').append(button_labels);

    // Populate supplemental categories
    var suppl_question_html = '';
    instr_table_entries = '';
    for (i=0; i<supplemental_questions.length; i++) {
        var groupId = supplemental_questions[i][0],
            groupName = supplemental_questions[i][1],
            autosized = supplemental_questions[i][2],
            options = supplemental_questions[i][3];
        var question_group = '<div class="form-group"><div id="' + groupId + '" class="btn-group btn-group-justified' + (autosized ? ' btns-autosized' : '') + '" style="display: none;" data-toggle="buttons">'

        var table_headers = ''
        var table_descs = ''
        for (j=0; j<options.length; j++) {
            var optionName = options[j][0],
                optionTitle = options[j][1],
                optionDesc = options[j][2];
            question_group += '<label class="btn btn-xs btn-default" data-toggle="tooltip" title="' + optionDesc + '"><input name="' + groupName + '" type="radio" value="' + optionName + '" />' + optionTitle + '</label>';
            table_headers += '<th>' + optionTitle + '</th>';
            table_descs +=  '<td>' + optionDesc + '</td>';
        }

        question_group += '</div></div>\n';
        suppl_question_html += question_group;
        instr_table_entries += '<table class="table table-condensed table-striped table-bordered table-responsive"><tr>' + table_headers + '</tr><tr>' + table_descs + '</tr></table>';
    }
    $('#SupplTable').append(instr_table_entries);
    $(suppl_question_html).insertAfter($('#CategoryGroup'));

    // Instructions expand/collapse
    var content = $('#instructionBody');
    var trigger = $('#collapseTrigger');
    trigger.click(function(){
        content.toggle();
        trigger.toggleClass('expanded');
    });

    // Button change handler
    $("#CategoryInputs input:radio").change(function() {
        // Highlight selected category
        $("#CategoryInputs input:radio").parent().removeClass("btn-primary");
        $("#CategoryInputs input:radio").parent().addClass("btn-default");
        if ($(this).is(":checked")){
            $(this).parent().removeClass("btn-default");
            $(this).parent().addClass("btn-primary");
        }
        // Manage supplemental questions
        if (['elections_hard', 'elections_soft', 'business_economics', 'government', 'current_events', 'cultural'].indexOf(this.id) >= 0){
            activateQuestion("#FactOpinionInputs");
            if ($("#FactOpinionInputs input:checked").val() == "fact"){
                activateQuestion("#InvestigativeInputs");
            }
        }else{
            deactivateQuestion("#FactOpinionInputs");
            deactivateQuestion("#InvestigativeInputs");
        }
        if (['business_economics', 'government', 'current_events', 'sports', 'cultural'].indexOf(this.id) >= 0){
            activateQuestion("#UsForeignInputs");
        }else{
            deactivateQuestion("#UsForeignInputs");
        }
        if (['elections_hard', 'elections_soft', 'business_economics', 'science_tech', 'government', 'entertainment', 'sports', 'products', 'anecdotes', 'current_events', 'cultural'].indexOf(this.id) >= 0){
            activateQuestion("#ToneInputs");
        }else{
            deactivateQuestion("#ToneInputs");
        }
        if (['elections_hard', 'elections_soft', 'business_economics', 'science_tech', 'government', 'entertainment', 'anecdotes', 'current_events', 'cultural'].indexOf(this.id) >= 0){
            activateQuestion("#EmotionInputs");
        }else{
            deactivateQuestion("#EmotionInputs");
        }
        // Show supplemental question label only if needed
        if (['none','transitions','ads','weather'].indexOf(this.id)>=0){
            $('#Question2label').remove();
        }else{
            if ($('#Question2label').length == 0) {
                $('<label id="Question2label" class="group-label">Pick the best choice in each group:</label>').insertAfter('#CategoryGroup');
            }
        }
    });
    // Only show investigative question if 'fact' is selected
    $("#FactOpinionInputs input:radio").change(function() {
        if (this.value == 'fact'){
            activateQuestion("#InvestigativeInputs");
        }else{
            deactivateQuestion("#InvestigativeInputs");
        }
    });
});