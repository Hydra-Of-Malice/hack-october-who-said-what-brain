import { useCallback } from "react";
import { useNavigate } from "react-router-dom";
import { requestSeek, type SeekRequest } from "../../lib/player";

export function meetingPath(meetingId: string): string {
  return `/m/${encodeURIComponent(meetingId)}`;
}

/** Seek the meeting player to a moment, opening the meeting first when another page is showing. */
export function useJump(): (r: SeekRequest) => void {
  const navigate = useNavigate();
  return useCallback(
    (r: SeekRequest) => {
      requestSeek(r);
      const path = meetingPath(r.meeting_id);
      // Read the live location: callers may run after an await, when the route has moved on.
      if (window.location.pathname !== path) navigate(path);
    },
    [navigate],
  );
}

/** Open a meeting without seeking. */
export function useOpenMeeting(): (meetingId: string) => void {
  const navigate = useNavigate();
  return useCallback(
    (meetingId: string) => {
      const path = meetingPath(meetingId);
      if (window.location.pathname !== path) navigate(path);
    },
    [navigate],
  );
}
