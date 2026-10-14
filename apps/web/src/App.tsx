import { BrowserRouter, Navigate, Route, Routes } from "react-router-dom";
import AppShell from "./components/shell/AppShell";
import AskDrawer from "./components/ask/AskDrawer";
import CommandPalette from "./components/CommandPalette";
import MeetingsPage from "./pages/MeetingsPage";
import MeetingRoom from "./pages/MeetingRoom";
import CommitmentsPage from "./pages/CommitmentsPage";
import DecisionsPage from "./pages/DecisionsPage";
import PeoplePage from "./pages/PeoplePage";
import GraphPage from "./pages/GraphPage";

export default function App() {
  return (
    <BrowserRouter>
      <AppShell>
        <Routes>
          <Route path="/" element={<MeetingsPage />} />
          <Route path="/m/:id" element={<MeetingRoom />} />
          <Route path="/commitments" element={<CommitmentsPage />} />
          <Route path="/decisions" element={<DecisionsPage />} />
          <Route path="/people" element={<PeoplePage />} />
          <Route path="/graph" element={<GraphPage />} />
          <Route path="*" element={<Navigate to="/" replace />} />
        </Routes>
      </AppShell>
      <AskDrawer />
      <CommandPalette />
    </BrowserRouter>
  );
}
