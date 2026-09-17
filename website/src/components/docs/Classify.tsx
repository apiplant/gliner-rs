import { DocsLayout } from "./DocsLayout";
import { H1, H2, H3, Lead, P, IC, Pre, FlagTable, Section } from "./Prose";
import { CopyBlock } from "../Code";

export function DocsClassify() {
  return (
    <DocsLayout>
      <H1><IC>gliner-classify</IC></H1>
      <Lead>
        Zero-shot text classification straight from command-line flags, or through an interactive
        shell that loads the model once and keeps history.
      </Lead>

      <Section>
        <H2>Build</H2>
        <CopyBlock command="cargo build --release                      # add --features cuda for GPU" />
        <P>
          <IC>--model-variant {"{"}multi,small,base{"}"}</IC> picks between{" "}
          <IC>fastino/gliner2.5-multi-v1</IC> (default, 205M, all languages),{" "}
          <IC>fastino/gliner2.5-small-v1</IC> and <IC>fastino/gliner2.5-base-v1</IC> (smaller,
          faster, English-leaning). Each downloads automatically into the cache directory on first
          use.
        </P>
      </Section>

      <Section>
        <H2>One-shot CLI</H2>
        <P>Texts come from positional arguments, from <IC>-f file</IC> (one per line), or from piped stdin.</P>

        <H3>Single-label</H3>
        <CopyBlock command={`gliner-classify -l positive,negative,neutral "I love this phone"`} />
        <Pre>{`label: positive (1.000)`}</Pre>

        <H3>Multi-label</H3>
        <P><IC>-m</IC>; every label at or above <IC>--threshold</IC> (default 0.5).</P>
        <CopyBlock
          command={`gliner-classify -m -l camera,performance,battery,display,price \\
    "Great camera quality, decent performance, but poor battery life."`}
        />
        <Pre>{`label: camera (0.658), performance (0.790), battery (0.526)`}</Pre>

        <H3>Several tasks in one pass</H3>
        <P>
          <IC>+</IC> makes a task multi-label and <IC>-a</IC> shows every probability (
          <IC>*</IC> marks the predictions).
        </P>
        <CopyBlock
          command={`gliner-classify -t sentiment=positive,negative -t +topics=technology,sports,politics,finance -a \\
    "The Fed raised rates, and tech stocks tumbled."`}
        />
        <Pre>{`sentiment: *negative (1.000), positive (0.000) | topics: *finance (0.984), *technology (0.898), *politics (0.542), sports (0.001)`}</Pre>

        <H3>Batch input with a prompt</H3>
        <P>Each line of output is the text, a tab, then the result.</P>
        <CopyBlock
          command={`gliner-classify -l book_flight,cancel_booking,check_status,baggage_info,talk_to_human \\
    -p "What does the customer want to do?" \\
    "My flight to Rome got moved, can I get my money back?" \\
    "where is my suitcase" \\
    "just give me a real person please"`}
        />
        <Pre>{`My flight to Rome got moved, can I get my money back?\tlabel: cancel_booking (0.721)
where is my suitcase\tlabel: baggage_info (0.993)
just give me a real person please\tlabel: talk_to_human (1.000)`}</Pre>

        <H3>Multilingual</H3>
        <P>English labels work on any language.</P>
        <CopyBlock
          command={`gliner-classify -l positive,negative,neutral "Das Essen war kalt und der Kellner unhöflich." \\
    "这家餐厅的服务太棒了！" "C'était correct, sans plus."`}
        />
        <Pre>{`Das Essen war kalt und der Kellner unhöflich.\tlabel: negative (0.857)
这家餐厅的服务太棒了！\tlabel: positive (1.000)
C'était correct, sans plus.\tlabel: positive (0.724)`}</Pre>

        <H3>Descriptions and few-shot examples</H3>
        <P>With <IC>-k N</IC> for the top N labels.</P>
        <CopyBlock
          command={`gliner-classify -t "queue=billing:Payments invoices refunds,tech:Bugs crashes errors,account:Login password access" \\
     -t priority=urgent,normal,low \\
     -e "Production is down for all users!!=>urgent" -k 2 \\
     "I was charged twice this month and can't log in to fix it"`}
        />
        <Pre>{`queue: *billing (0.960), account (0.029) | priority: *urgent (0.895), normal (0.064)`}</Pre>

        <H3>Pipelines</H3>
        <P>With <IC>--format jsonl|json|tsv</IC>.</P>
        <CopyBlock command={`printf "Win a free iPhone now\\nLunch tomorrow?\\n" | gliner-classify -l spam,ham --format tsv`} />
        <Pre>{`Win a free iPhone now\tlabel\tspam\t0.6519
Lunch tomorrow?\tlabel\tham\t0.5609`}</Pre>
        <CopyBlock
          command={`gliner-classify -t "+flags=toxic:Insults or harassment,spam:Ads or scams,nsfw:Sexual content,self_harm" \\
    --threshold 0.4 --format jsonl "Click here to win a free iPhone, you idiot"`}
        />
        <Pre lang="json">{`{"text":"Click here to win a free iPhone, you idiot","flags":{"labels":["toxic","nsfw"],"confidences":[0.7689375877380371,0.6499032974243164]}}`}</Pre>

        <P>
          Zero-shot labels and descriptions are prompts, and wording matters. In the last example
          the model flags <IC>nsfw</IC> rather than <IC>spam</IC>, so tune the descriptions and
          threshold on your own data. Adding descriptions can even flip a result: for{" "}
          <IC>spam,ham</IC> on "Win a free iPhone now, click here!", bare labels give{" "}
          <IC>spam</IC> and the descriptions above give <IC>ham</IC>.
        </P>
      </Section>

      <Section>
        <H2>Flags</H2>
        <FlagTable
          rows={[
            { flag: "-l, --labels a,b:desc,c", meaning: "labels of the default task (named by -n, default label)" },
            { flag: "-m, --multi", meaning: "make the --labels task multi-label" },
            { flag: "-t, --task [+]name=a,b", meaning: "add a task (+ = multi-label), repeatable" },
            { flag: '-e, --example "text=>label"', meaning: "few-shot example, used by tasks that have the label" },
            { flag: "-p, --prompt TEXT", meaning: "instruction appended to the task name" },
            { flag: "--threshold 0.5", meaning: "multi-label cutoff" },
            { flag: "--activation auto|softmax|sigmoid", meaning: "override scoring (auto: softmax single, sigmoid multi)" },
            { flag: "-a, --all / -k, --top-k N", meaning: "show all or the top N label probabilities" },
            { flag: "--format text|jsonl|json|tsv", meaning: "output format" },
            { flag: "-f, --file PATH", meaning: "read texts line by line" },
            { flag: "-i, --interactive", meaning: "start the shell" },
            { flag: "--model DIR, --cuda, --fp16, -v", meaning: "model location, device, precision, timing" },
          ]}
        />
      </Section>

      <Section>
        <H2>Interactive shell</H2>
        <P>
          Run <IC>gliner-classify -i</IC>, or run <IC>gliner-classify</IC> with no input in a
          terminal. Preload settings with the usual flags, e.g.{" "}
          <IC>gliner-classify -i -l positive,negative</IC>. The model loads once; after that,
          anything you type that isn't a command is classified with the current settings.
        </P>
        <Pre caption="gliner-classify -i" lang="console">{`$ gliner-classify -i
loading model from .. ...
model loaded in 344.93ms
gliner-classify shell: type text to classify, :help for commands, Ctrl-D to quit
classify> :labels positive,negative,neutral
label> :task +topics=battery,camera,shipping,price,display
[2 tasks]> :all
[2 tasks]> Arrived late and the camera is blurry
label: *negative (0.991), positive (0.005), neutral (0.004) | topics: *camera (0.764), shipping (0.217), display (0.134), price (0.055), battery (0.007)`}</Pre>
        <P>
          The prompt shows what's active: <IC>label{">"}</IC> for one task, <IC>+topics{">"}</IC>{" "}
          for a multi-label task, and <IC>[2 tasks]{">"}</IC> for several.
        </P>

        <H3>Recipes</H3>

        <P><strong class="text-ink font-medium">Emotion wheel, top 3</strong></P>
        <Pre lang="console">{`classify> :labels joy,sadness,anger,fear,surprise,disgust,trust,anticipation
label> :top 3
label> I can't believe they actually picked my design for the launch!
label: *surprise (0.762), trust (0.092), anticipation (0.052)`}</Pre>

        <P><strong class="text-ink font-medium">Support ticket router</strong></P>
        <Pre lang="console">{`classify> :task queue=billing:Payments invoices refunds,tech:Bugs crashes errors,account:Login password access
queue> :task priority=urgent,normal,low
[2 tasks]> :example Production is down for all users!!=>urgent
[2 tasks]> :top 2
[2 tasks]> I was charged twice this month and can't log in to fix it
queue: *billing (0.960), account (0.029) | priority: *urgent (0.895), normal (0.064)`}</Pre>

        <P><strong class="text-ink font-medium">Chatbot intent detection</strong></P>
        <Pre lang="console">{`classify> :labels book_flight,cancel_booking,check_status,baggage_info,talk_to_human
label> :prompt What does the customer want to do?
label> where is my suitcase
label: baggage_info (0.993)
label> just give me a real person please
label: talk_to_human (1.000)`}</Pre>

        <P><strong class="text-ink font-medium">Moderation as JSON</strong></P>
        <Pre lang="console">{`classify> :task +flags=toxic:Insults or harassment,spam:Ads or scams,nsfw:Sexual content,self_harm
+flags> :threshold 0.4
+flags> :format jsonl
+flags> Click here to win a free iPhone, you idiot
{"text":"Click here to win a free iPhone, you idiot","flags":{"labels":["toxic","nsfw"],"confidences":[0.7689375877380371,0.6499032974243164]}}`}</Pre>

        <P><strong class="text-ink font-medium">Batch a file</strong></P>
        <Pre lang="console">{`classify> :labels spam,ham
label> :format tsv
label> :time
label> :file ~/mail/inbox.txt
Win a free iPhone now\tlabel\tspam\t0.6519
Lunch tomorrow?\tlabel\tham\t0.5609
(2 text(s) in …)`}</Pre>

        <H3>Commands</H3>
        <FlagTable
          rows={[
            { flag: ":labels a,b:desc,c", meaning: "set the default label task" },
            { flag: ":multi [on|off]", meaning: "toggle multi-label on the label task" },
            { flag: ":task [+]name=a,b / :rm name", meaning: "add or replace a task / remove it" },
            { flag: ":example text=>label / :example clear", meaning: "add few-shot examples / drop them" },
            { flag: ":prompt TEXT|off", meaning: "set the instruction" },
            { flag: ":threshold 0.4, :activation softmax", meaning: "scoring" },
            { flag: ":all [on|off], :top N|off", meaning: "show probabilities" },
            { flag: ":format text|jsonl|json|tsv", meaning: "output format" },
            { flag: ":file PATH", meaning: "classify each line of a file" },
            { flag: ":time [on|off]", meaning: "print timing after each run" },
            { flag: ":show, :clear, :help, :quit", meaning: "inspect, reset, help, exit" },
          ]}
        />

        <P>
          <strong class="text-ink font-medium">Tips:</strong> <IC>:</IC> followed by Tab completes
          commands. Up arrow and Ctrl-R search history, which persists in{" "}
          <IC>$XDG_STATE_HOME/gliner-classify/history</IC> (default{" "}
          <IC>~/.local/state/gliner-classify/history</IC>). Ctrl-C clears the current line and
          Ctrl-D exits. Enter one command per line: pasting several lines at once can drop some of
          them. For snappy demos, start with <IC>--cuda --fp16</IC> and turn on <IC>:time</IC>. On
          CPU a short text takes about 0.2s.
        </P>
      </Section>
    </DocsLayout>
  );
}
