import { useCallback, useEffect, useState } from "react";

import { AppHeader } from "./components/AppHeader";
import { ChatList } from "./components/ChatList";
import { Composer } from "./components/Composer";
import { Inspector } from "./components/Inspector";
import { LanguageTab } from "./components/LanguageTab";
import { MessageList } from "./components/MessageList";
import { PrivacyTab } from "./components/PrivacyTab";
import { SourcesTab } from "./components/SourcesTab";
import { useTheme } from "./hooks/useTheme";
import { deleteChat, getChat, getHealth, getMode, listChats, setMode } from "./lib/api";
import type { ChatSummary, Health, Mode, ModelName } from "./lib/types";
import { useChat } from "./state/useChat";
import { useEgressLog } from "./state/useEgressLog";

export default function App() {
  const { theme, toggle } = useTheme();
  const chat = useChat();
  const [health, setHealth] = useState<Health | null>(null);
  const [healthError, setHealthError] = useState(false);
  const [mode, setModeState] = useState<Mode | null>(null);
  const [chats, setChats] = useState<ChatSummary[]>([]);
  const [model, setModel] = useState<ModelName>("ollama");
  const [override, setOverride] = useState<string | null>(null);
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [sheetOpen, setSheetOpen] = useState(false);
  const [modeBusy, setModeBusy] = useState(false);
  const { entries, failed: egressLogFailed } = useEgressLog(true);

  const refreshChats = useCallback(async () => {
    try {
      setChats(await listChats());
    } catch {
      // The list is a convenience; a failure here must not stop the chat working.
    }
  }, []);

  useEffect(() => {
    void (async () => {
      try {
        setHealth(await getHealth());
        setHealthError(false);
        setModeState(await getMode());
      } catch {
        setHealth(null);
        setHealthError(true);
      }
      await refreshChats();
    })();
  }, [refreshChats]);

  const turns = chat.state.turns;
  const selected = turns.find((turn) => turn.id === selectedId) ?? turns[turns.length - 1] ?? null;
  const answerModel: ModelName = mode?.private === true ? "ollama" : model;

  async function onSend(question: string) {
    await chat.ask(question, answerModel, override);
    await refreshChats();
  }

  async function onPrivateChange(isPrivate: boolean) {
    setModeBusy(true);
    try {
      setModeState(await setMode(isPrivate));
      chat.reset();
      setSelectedId(null);
      await refreshChats();
    } catch {
      // A 409 means a switch is already running; the state stays as the backend last reported it.
    } finally {
      setModeBusy(false);
    }
  }

  async function onOpenChat(chatId: string) {
    try {
      chat.load(await getChat(chatId));
      setSelectedId(null);
    } catch {
      // A chat deleted in another window is simply gone; the list is refreshed below.
      await refreshChats();
    }
  }

  async function onDeleteChat(chatId: string) {
    try {
      await deleteChat(chatId);
    } finally {
      if (chat.state.chatId === chatId) {
        chat.reset();
      }
      await refreshChats();
    }
  }

  const panels = [
    {
      id: "language",
      label: "Language",
      content: (
        <LanguageTab language={selected?.language ?? null} override={override} onOverride={setOverride} />
      ),
    },
    {
      id: "sources",
      label: "Sources",
      content: (
        <SourcesTab
          sources={selected?.sources ?? []}
          web={selected?.web ?? []}
          citations={selected?.citations ?? null}
        />
      ),
    },
    {
      id: "privacy",
      label: "Privacy",
      content: (
        <PrivacyTab
          mode={mode}
          entries={entries}
          entriesFailed={egressLogFailed}
          busy={modeBusy}
          onChange={onPrivateChange}
        />
      ),
    },
  ];

  return (
    <div className="flex h-full flex-col bg-paper">
      <AppHeader
        health={health}
        healthError={healthError}
        mode={mode}
        model={model}
        busy={modeBusy}
        theme={theme}
        onModelChange={setModel}
        onPrivateChange={onPrivateChange}
        onToggleTheme={toggle}
      />
      <div className="grid min-h-0 flex-1 grid-cols-1 workspace:grid-cols-[16rem_1fr_22rem]">
        <div className="hidden workspace:block">
          <ChatList
            chats={chats}
            currentId={chat.state.chatId}
            isPrivate={mode?.private ?? false}
            onNew={() => {
              chat.reset();
              setSelectedId(null);
            }}
            onSelect={(id) => void onOpenChat(id)}
            onDelete={(id) => void onDeleteChat(id)}
          />
        </div>
        <main className="flex min-h-0 flex-col">
          <div className="min-h-0 flex-1 overflow-y-auto">
            <MessageList
              turns={turns}
              selectedId={selected?.id ?? null}
              onSelect={(id) => {
                setSelectedId(id);
                setSheetOpen(true);
              }}
              onRetry={() => void chat.retry(answerModel, override)}
            />
          </div>
          <div className="flex items-center justify-end px-4 pt-2 workspace:hidden">
            <button
              type="button"
              onClick={() => setSheetOpen((open) => !open)}
              className="rounded border border-line px-2 py-1 text-xs text-muted"
            >
              Details
            </button>
          </div>
          <Composer
            onSend={(question) => void onSend(question)}
            onStop={chat.stop}
            streaming={chat.streaming}
            disabled={health === null || chat.streaming}
          />
        </main>
        <aside
          data-testid="inspector-region"
          data-open={sheetOpen ? "true" : "false"}
          className={
            sheetOpen
              ? "fixed inset-x-0 bottom-0 z-10 max-h-[70vh] border-t border-line pb-[env(safe-area-inset-bottom,0px)] workspace:static workspace:max-h-none workspace:border-l workspace:border-t-0 workspace:pb-0"
              : "hidden workspace:block workspace:border-l workspace:border-line"
          }
        >
          <Inspector panels={panels} onClose={sheetOpen ? () => setSheetOpen(false) : undefined} />
        </aside>
      </div>
    </div>
  );
}
