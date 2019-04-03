'''
Baseline script for the Semeval 2019 Task 9 competition.
Written by Sapna Negi, Tobias Daudert
Contact: sapna.negi@insight-centre.org, tobias.daudert@insight-centre.org
Input:  test_data_for_subtaskA.csv which has to be in the same folder as this script. The name of the csv.-file is not fixed. The accepted structure is: id, sentence, prediction
        test_data_for_subtaskB.csv which has to be in the same folder as this script. The name of the csv.-file is not fixed. The accepted structure is: id, sentence, prediction
Output: test_data_for_subtaskA_predictions.csv, test_data_for_subtaskB_predictions.csv
Run the script: semeval-task9-baseline.py test_data_for_subtaskA_predictions.csv test_data_for_subtaskB_predictions.csv
'''

import nltk
import csv
import re
import sys
from nltk.tokenize import word_tokenize

import Preprocessing

trainingData = "V1.4_Training_new.csv"
testData = "SubtaskB_EvaluationData_labeled.csv"

# To be uncommented in case the average_perceptron_tagger is not on your machine yet
# import nltk
# nltk.download('averaged_perceptron_tagger')

data_path = trainingData
data_path2 = testData
out_path = trainingData[:-4] + "_predictions2.csv"
out_path2 = testData[:-4] + "_VLCGA-NB.csv"


class taggingParsing:

    def sentenceSplit(self, text):
        tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
        sentences = tokenizer.tokenize(text)
        return sentences

    def taggingNLTK(self, text):
        tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
        sentences = tokenizer.tokenize(text)
        for sent in sentences:
            text = word_tokenize(sent)
            tagged_sent = nltk.pos_tag(text)


import pandas as pd


def readFile(file):
    df = pd.read_csv(file)
    df.head()

    # add a column encoding the label as an integer
    # because categorical variables are often better represented by integers than strings
    col = ['label', 'sentence']
    df = df[col]
    df = df[pd.notnull(df['sentence'])]
    df.columns = ['label', 'sentence']
    df['category_id'] = df['label']
    df['label_str'] = df['label'].replace({1: 'suggestion', 0: 'non-suggestion'})
    # preprocessing training data
    contractions = Preprocessing.loadCSV('Contractions.csv', 'contraction', 'text1').to_dict('split')
    df = Preprocessing.Preprocessing(df, contractions)
    df.head()
    return df


def classify(sent_list):
    suggestionKeywords = ["suggest", "recommend", "hopefully", "go for", "request", "it would be nice", "adding",
                "should come with", "should be able", "could come with", "i need", "we need", "needs", "would like to",
                "would love to", "allow", "add"]

    # keywords obtained from VLCGA-NB
    suggestionKeywords = ["unnecessarily","afterwards","predictive","revocation","adversely","torch","specialized","viewpoint","param","flag","mediation","freezing","expiry","premium","downgrade","glass","adopter","invent","scholarship","pertinent","refined","pitch","rip","declared","sky","dismiss","comfortable","analyzer","noise","heat","transmission","mathematical","interrogate","descriptive","legacy","drainage","nerve","grammatical","leg","revise","prove","unlucky","personalization","influence","hobbyist","nullable","photograph","resumable","safely","trough","excuse","inserted","unhappy","augmented","jet","sudden","consolidate","balanced","changer","seated","trailer","clap","hamstring","uninteresting","unavailable","plea","alienate","untangle","mirror","whistle","presume","commuter","pandora","precise","consume","bottleneck","cycling","cortina","engagement","currency","affiliate","easing","impacted","petition","die","sibling","seemless","gif","gode","misleading","hook","unregistered","logically","skipping","projector","regulated","ribbon","flowing","sacrificing","collaborative","discoverability","odd","flashlight","vibration","colorful","hurting","elsewhere","wildly","cap","surname","someday","participate","coax","penetrating","permanently","escape","outline","mover","dimension","quarter","inclusion","marker","lingual","frustum","throne","earnings","clothes","jarring","connectable","withdrawal","alleviate","delta","cooking","latex","mole","unwieldy","exceeding","pronoun","spec","whereby","mitigate","accessibility","navigational","patch","renderer","looping","guest","indigo","granular","cleaning","lettering","outer","mutual","air","tilt","decompress","disastrously","technique","haptic","aim","nudity","plumbing","assistant","bright","finer","authority","accomplished","surrounded","geo","beautifully","join","protection","writeable","warn","mile","eliminate","elevate","freely","offense","alter","slip","locker","rogue","admirable","batch","broad","participation","accepted","ambiguous","unselected","erroneous","ingredient","aint","attractive","erase","disappointing","tiled","ruin","approval","nutshell","eagerly","technically","recoup","frontal","awkward","excerpt","proportion","disposable","yearly","externally","scree","spit","incompatibility","authorize","anaglyph","composed","dup","segregate","sand","apostrophe","aging","dot","diversify","belief","deactivate","scrolled","stringent","initialize","virtualize","hopeless","exempt","feasible","vacuum","preferable","pursue","optimize","philosophical","insight","shall","meaningful","relational","sideways","exclamation","intransparent","autonomous","layered","tiny","granularly","deletion","outcome","creat","expedited","happier","dominated","selective","shortener","combining","artificially","brand","emulation","plumb","refund","voodoo","crossing","referrer","elegant","watcher","highest","ban","underhanded","overload","final","achievable","sortable","toolbox","removal","dropper","utilization","soul","expander","chatting","circle","urgently","compressed","prominently","dimensional","schema","film","prey","wasnt","decimal","wisdom","wether","asynchronous","relate","bash","guarding","traversing","extent","employee","weed","validate","piping","proceeding","grip","proportionally","analyse","regionally","guarantee","onwards","green","backward","unabstract","tuner","trash","plotted","minimally","equally","outgoing","revoke","expensive","observable","undo","employed","whoever","simplistic","unsaved","rename","serialize","zipping","affair","hood","alignment","acrylic","cough","reschedule","inheritance","lesser","dedo","shortcoming","stilted","fantasy","holly","unfinished","consent","foo","lys","soccer","semantically","vendor","draft","brokerage","visa","conditionally","pressure","compliment","strip","clog","duplication","prefix","plane","reflecting","cope","simplicity","evidently","throwaway","sliding","retain","dump","subset","demonstrate","suggesting"]
    nonSuggestionKeywords = ["want", "an", "so", "zune", "hi", "setting", "introduced", "generated", "cell", "manually",
                             "loop", "tried", "think", "provide", "myself", "deleting", "faster", "lets", "perhaps",
                             "matching", "usually", "messaging", "idea", "writes", "days", "sign", "appear",
                             "animations", "microsoft", "loaded", "title", "our", "event", "header", "ink", "each",
                             "mac", "search", "player", "force", "community", "scrollviewer", "saw", "between",
                             "choice", "smartcard", "perform", "save", "notification", "possible", "weeks", "emulator",
                             "followed", "extensions", "emulate", "fine", "truly", "good", "cannot", "you", "connected",
                             "certainly", "text-align:", "future", "there", "(or", "operation", "sometimes", "specific",
                             "course", "took", "windows", "removed", "certificate", "normal", "line", "has",
                             "submitted", "opens", "also", "area", "expired", "crashes", "phone", "getting", "heard",
                             "said", "world", "even", "im", "users", "containing", "sensitive", "new", "plays",
                             "payment", "handler", "make", "impossible", "action", "times", "after", "even", "headers",
                             "least", "down", "devices", "last", "in-app","places", "id", "cortana", "total", "leaving", "onto", "develop", "second", "clear", "what", "connect", "high", "follow", "unit", "settings", "when", "supported", "matter", "lastname", "icons", "all", "worse", "reports", "hear", "including", "easily", "publishing", "internet", "again", "separate", "great", "dynamic", "while", "testing", "metadata", "two", "let", "whatever", "side", "files", "currently", "radio", "calendarview", "love", "creation", "running", "sms", "happy", "fileopenpicker", "trial", "price", "tumblr", "mobile", "http", "certification", "isolated", "drop", "embedded", "couldnt", "linux", "reset", "portal", "udp", "preferred", "make", "people", "hub", "certificates", "package", "disk", "localization", "reach", "hls", "tie", "signed", "features", "animations", "toggle", "three", "supports", "problem", "app", "author", "depth", "week", "functions", "effect", "complex", "safari", "which", "websites", "via", "once", "here", "commands", "ways", "password", "private", "limited", "users", "attach", "variations", "submit", "answer", "sql", "panorama", "is", "uploaded", "pressed", "time	", "some", "frame", "manager", "lots", "leave", "behavior", "real", "it", "same", "serialization", "google", "full", "here:", "async", "initial", "sample", "passed", "dropping", "certain", "like", "month", "games", "add", "them", "and", "c++", "generate", "windows", "folders", "youre", "string", "music", "kit", "hyper-v", "limitations", "simple", "formats", "who", "entire", "win", "replace", "later", "vision", "graphics", "not", "while", "configure", "gives", "document", "ive", "status", "definitely", "brightness", "battery", "audio", "longer", "cases", "date", "nearly", "sorting", "how", "to", "calendar", "microsoft", "these", "bring", "feedback", "experience", "unfortunately", "quickly", "typing", "from", "users", "focused", "iot", "updated", "outside", "according", "house", "third", "control", "anyone", "play", "webview", "xbox", "perfectly", "exposed", "assume", "accessible", "programming", "idle", "subscribe", "environment", "storage", "add", "say", "obvious", "else", "going", "safe", "color", "(as", "limits", "example", "packages", "easier", "whether", "computer", "", "trigger", "enough", "that", "above", "when", "someone", "ideal", "prevent", "reporting", "tag", "switch", "run", "beyond", "usb", "installed", "form", "forces", "notice", "this", "works", "sim", "fast", "front", "point", "export", "strange", "lines", "im", "true", "process", "send", "protocol", "created", "included", "however", "then", "chose", "happen", "(e.g", "determine", "applications", "stay", "controls", "server", "what", "all", "nfc", "default", "selecting", "classic", "working", "impact", "seems", "smart", "marketplace", "button", "center", "contacts", "stored", "chain", "unless", "things", "more", "without", "present", "recommended", "implementation", "great", "since", "was", "try", "numbers", "more", "thing", "drag", "to", "isolation", "huge", "subscribed", "something", "exists", "set", "samples", "read", "marked", "docs", "ridiculous", "solution", "available", "cause", "end", "mouse", "ios", "icon", "became", "configuration", "suggest", "finguerprint", "independent", "app", "publish", "connection", "developed", "multiple", "about", "entry", "here", "call", "ask", "that", "simply", "website", "fields", "i", "makes", "(like", "in", "bugs", "later", "listview", "paste", "tasks", "listed", "details", "suggestedstartlocation", "xamarin.forms", "posting", "videos", "fixed", "bad", "countries", "next", "applications", "google", "enterprise", "automatic", "call", "apps", "absolutely", "like", "windows", "dealing", "test", "reduce", "script", "interface", "choices", "recording", "minute", "actual", "buy", "allowing", "unique", "site", "interesting", "pain", "reviews", "standard", "example:", "device", "phones", "duplicate", "feature", "wont", "lets", "you", "please", "unable", "tablet", "excel", "app", "important", "cellular", "below", "but", "do", "touch", "scenarios", "dedicated", "card", "ready", "to", "taken", "preferredlaunchviewsize", "help", "views", "these", "contains", "finger", "and", "best", "apis", "itself", "instance", "development", "downloads", "dialog", "under", "compile", "objects", "far", "route", "function", "day", "library", "care", "deep", "helpful", "automatically", "service", "team", "issue:", "upgrade", "feedly", "var", "didnt", "soft", "auto", "arm", "nothing", "internal", "options", "option", "invoice", "physical", "reader", "job", "when", "renaming", "properties", "xap", "hope", "shouldnt", "add-on", "quick", "debug", "behavior", "showing", "the", "probably", "loading", "are", "tools", "bluetooth", "post", "memory", "position", "tap", "currently", "successful", "more", "hardware", "thank", "for", "either", "amount", "third-party", "dark", "inside", "correctly", "lose", "limitation", "german", "name", "software", "com", "skip", "older", "handle", "whole", "(even", "blocking", "generates", "functionality", "keeps", "purchase", "regarding", "azure", "selection", "driver", "building", "forced", "host", "bar", "etc", "sends", "display", "paid", "mark", "bars", "did", "completely", "remove", "official", "restart", "version", "files", "vote", "publisher", "almost", "operator", "class", "cloud", "and/or", "related", "at", "quality", "highly", "communicate", "tracking", "dialogs", "starting", "choose", "requesting", "half", "using", "reveal", "near", "they", "light", "smartphone", "retrieve", "viewer", "center", "frameworks", "financial", "exception", "devs", "itself", "let", "website", "tags", "difficult", "htc", "voice", "a", "exchange", "processing", "subject", "marketplace", "experience", "arent", "midi", "right", "nokia", "money", "opened", "feel", "better", "lock", "she", "playing", "active", "match", "flow", "seconds", "languages", "popup", "transition", "twitter", "launched", "surface", "operations", "placed", "characters", "ago", "moving", "wide", "have", "same", "articles", "nor", "how", "user", "apply", "value", "binding", "requires", "swipe", "moment", "sure", "solution", "manifest", "customer", "bought", "allow", "hours", "this", "life", "black", "verify", "activate", "extra", "soon", "done", "microsoft", "bridge", "location", "wanted", "contentdialog", "navigate", "folder", "relevant", "api", "only", "worked", "store", "advertising", "design", "id", "rendering", "confusing", "as", "scroll", "una", "shows", "built", "addition", "accounts", "search", "enabled", "submissions", "hide", "could", "invoke", "codes", "generic", "capable", "messy", "entries", "zoom", "improve", "instapaper", "number", "situation", "only", "pin", "throw", "thread", "unfortunately", "recently", "applications", "state", "pdfs", "trying", "px;", "etc", "thanks", "just", "well", "low", "stop", "uses", "registered", "map", "reading", "his", "template", "a", "exist", "input", "api", "account", "written", "causes", "on", "gets", "on", "applies", "hijri", "due", "window", "their", "drivers", "registration", "adjustment", "directory", "party", "code", "calender", "any", "open", "there", "alarm", "empty", "issue", "amazing", "link", "making", "cursor", "lot", "analytics", "causing", "app", "whats", "video", "theres", "e.g", "where", "year", "powerful", "several", "published", "app.", "particular", "right", "would", "here", "ive", "queue", "doing", "implement", "browser", "most", "ability", "times", "than", "slow", "time", "iap", "it", "common", "menu", "until", "fall", "or", "displayed", "loads", "thousands", "images", "that", "includes", "runtime", "age", "screenshot", "fail", "its", "apps", "user", "values", "for", "browser", "needed", "names", "takes", "cover", "framework", "tiles", "both", "through", "know", "complete", "calls", "iso", "animation", "usage", "edit", "they", "groups", "button", "purchased", "appears", "this", "first", "size", "wish", "onedrive", "change", "wait", "this", "native", "taskbar", "ones", "converter", "frustrating", "believe", "anything", "painful", "few", "transport", "reminder", "close", "returns", "giving", "some", "element", "adding", "attribute", "wrong", "catch", "rating", "tested", "application", "forum", "from", "scenario", "server", "order", "thereby", "past", "though", "known", "corporate", "performance", "one", "base", "many", "foreground", "per", "page", "pause", "pull", "(for", "report", "alone", ".net", "beta", "having", "white", "except", "notifications", "cant", "but", "into", "since", "restriction", "another", "buffering", "combobox", "wp", "would", "added", "gmail", "platforms", "points", "sort", "device", "customize", "see", "compiled", "desktop", "updates", "types", "wp.", "uwp", "feature", "maps", "is", "home", "lumia", "find", "scrollbar", "our", "issue", "reply", "msdn", "handled", "however", "events", "but", "this", "microphone", "after", "starts", "hand", "decided", "toolkit", "guys", "stuff", "reproduce", "stack", "one", "sorted", "history", "being", "rate", "can", "chinese", "effort", "please", "download", "raised", "allow", "visibility", "effects", "become", "evernote", "received", "receive", "poor", "requirements", "keys", "shared", "mode", "now", "cookies", "navigation", "and", "offer", "happening", "expected", "context", "manage", "hidden", "those", "goes", "screenshots", "error", "directly", "plugin", "potentially", "started", "out", "could", "maximum", "local", "keyword", "correctly", "general", "hello", "exact", "resolution", "port", "ring", "big", "security", "unlock", "mean", "explorer", "emulators", "searching", "knows", "sound", "sense", "result", "appx", "headers", "secondary", "developers", "upon", "before", "sometimes", "flyout", "smooth", "steps", "already", "changed", "it", "address", "microsofts", "news", "property", "device", "fine", "then", "creates", "actually", "capabilities", "mobile", "other", "progress", "phone", "prefer", "resource", "path", "old", "saved", "star", "blog", "had", "up", "key", "summary", "items", "especially", "fact", "splitview", "heavy", "reboot", "uwp", "put", "latest", "pressing", "custom", "revenue", "continue", "attached", "ideas", "now", "tapping", "build", "docker", "expect", "too", "resources", "benefits", "writing", "prompt", "format", "why", "style", "permission", "stores", "application", "dynamically", "disable", "space", "stopped", "components", "case", "given", "launcher", "fake", "networking", "support", "email", "idea", "etc", "now", "require", "workaround", "there", "happened", "prior", "android", "(not", "also", "account", "text", "went", "useful", "isnt", "credit", "single", "allowed", "serious", "such", "machine", "ringer", ".mp", "fixed", "log", "extension", "sounds", "select", "info", "original", "tooling", "been", "becomes", "potential", "windowsphone", "failing", "within", "share", "mouses", "existing", "textbox", "the", "somewhere", "page", "back", "limit", "enhance", "builds", "emulator", "accept", "dragging", "lacks", "cross", "platforms", "friends", "javascript", "android", "developer", "considering", "projects", "school", "readers", "colors", "review", "right", "", "click", "start", "thus", "part", "sets", "selected", "program", "none", "output", "at", "return", "tab", "able", "screen", "focus", "votes", "no", "resolve", "available", "made", "control", "await", "sources", "mechanism", "method", "code", "seeing", "project", "following", "creating", "further", "instead", "drive", "pc", "list", "access", "less", "regardless", "service", "deal", "define", "current", "wireless", "management", "allows", "print", "platform", "thats", "copy", "assembly", "from", "image", "searches", "increase", "free", "enable", "command", "told", "however", "just", "picker", "description", "enter", "not", "since", "ios", "well", "texture", "sdk", "smil", "wifi", "looks", "the", "screen", "svg", "min", "preview", "note", "update", "compatible", ")", "feedly", "apps", "many", "slider", "delete", "check", "may", "comes", "large", "links", "take", "caused", "and", "results", "firefox", "backrequested", "term", "capture", "convert", "pocket", "clicking", "media", "what", "means", "system", "changes", "filter", "previous", "that", "networks", "switching", "online", "using", "got", "possibility", "*", "updating", "failed", "customers", "based", "hard", "now", "buttons", "these", "feeds", "described", "reads", "scaling", "content", "own", "toolbar", "they", "show", "pass", "left", "other", "properly", "null", "used", "use", "me", "pivot", "in", "parameter", "provided", "triggered", "never", "place", "basic", "example)", "have", "broken", "built-in", "way", "work", "implemented", "theme", "hello", "code", "buffer", "restrictions", "will", "receipt", "assign", "reference", "paused", "my", "in", "iphone", "asking", "view", "photo", "login", "vertical", "services", "integrate", "then", "would", "give", "collection", "universal", "type", "nice", "report", "cards", "fix", "opening", "lack", "there", "computers", "commandbar", "looking", "every", "network", "register", "consumers", "youtube"]
    # Goldberg et al.
    pattern_strings = [r'.*would\slike.*if.*', r'.*i\swish.*', r'.*i\shope.*', r'.*i\swant.*', r'.*hopefully.*',
                       r".*if\sonly.*", r".*would\sbe\sbetter\sif.*", r".*should.*", r".*would\sthat.*",
                       r".*can't\sbelieve.*didn't.*", r".*don't\sbelieve.*didn't.*", r".*do\swant.*",
                       r".*i\scan\shas.*"]
    compiled_patterns = []
    for patt in pattern_strings:
        compiled_patterns.append(re.compile(patt))

    label_list = []
    for sent in sent_list:
        tokenized_sent = word_tokenize(sent[1])
        tagged_sent = nltk.pos_tag(tokenized_sent)
        tags = [i[1] for i in tagged_sent]
        label = 0
        patt_matched = False
        for compiled_patt in compiled_patterns:
            joined_sent = " ".join(tokenized_sent)
            matches = compiled_patt.findall(joined_sent)
            if len(matches) > 0:
                patt_matched = True
        suggestionKeyword_match = any(elem in suggestionKeywords for elem in tokenized_sent)
        nonsuggestionKeyword_match = any(elem in nonSuggestionKeywords for elem in tokenized_sent)
        pos_match = any(elem in ['MD', 'VB'] for elem in tags)

        # if patt_matched:
        #     label = 1
        # elif suggestionKeyword_match == True and nonsuggestionKeyword_match == False:
        #     label = 1
        # elif pos_match == True:
        #     label = 1
        # if nonsuggestionKeyword_match == True:
        #     label = 0
        # else:
        #     label=1
        if  pos_match == True:
            if suggestionKeyword_match == True or patt_matched:
                label = 1
        #
        # elif pos_match == True:
        #     label = 1

        label_list.append(label)

    from sklearn.metrics import confusion_matrix
    import seaborn as sns
    import matplotlib.pyplot as plt

    df = readFile(testData)
    category_id_df = df[['label_str', 'category_id']].drop_duplicates().sort_values('category_id')
    y_val = df.label
    conf_mat = confusion_matrix(y_val, label_list)
    sns.heatmap(conf_mat, annot=True, fmt='d',
                xticklabels=category_id_df.label_str.values, yticklabels=category_id_df.label_str.values)
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.show()

    from sklearn import metrics
    print(metrics.classification_report(y_val, label_list))
    return label_list


# This reads CSV a given CSV and stores the data in a list
def read_csv(data_path):
    file_reader = csv.reader(open(data_path, "rt", errors="ignore", encoding="utf-8"), delimiter=',')
    sent_list = []
    next(file_reader)
    for row in file_reader:
        id = row[0]
        sent = row[1]
        sent_list.append((id, sent))
    return sent_list


# This will create and write into a new CSV
def write_csv(sent_list, label_list, out_path):
    filewriter = csv.writer(open(out_path, "w+", newline='', encoding="utf-8"))
    count = 0
    for ((id, sent), label) in zip(sent_list, label_list):
        filewriter.writerow([id, sent, label])


if __name__ == '__main__':
    sent_list = read_csv(data_path2)
    label_list = classify(sent_list)
    write_csv(sent_list, label_list, out_path2)
