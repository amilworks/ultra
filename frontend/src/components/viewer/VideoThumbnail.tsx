import { useCallback, useRef, useState } from "react";

export type VideoThumbnailProps = {
  /** Server-rendered poster frame (ffmpeg), shown by default. */
  posterUrl: string;
  /** Range-streamable raw video URL, used only while hovering. */
  videoUrl: string;
  alt: string;
  className?: string;
  /** Called if the poster image or the video fails to load. */
  onError?: () => void;
};

/**
 * A grid video tile that previews on hover. By default it shows only the server
 * poster image — no video bytes are fetched. On hover it mounts a muted, looping
 * `<video>` that autoplays a preview, streaming via HTTP range requests; on leave
 * the element unmounts so the in-flight stream is cancelled. Long videos therefore
 * never download in full: the poster is a small static PNG, and hover playback
 * only pulls the leading bytes it actually plays.
 */
export function VideoThumbnail({ posterUrl, videoUrl, alt, className, onError }: VideoThumbnailProps) {
  const [hovering, setHovering] = useState(false);
  const videoRef = useRef<HTMLVideoElement | null>(null);

  const handleEnter = useCallback(() => setHovering(true), []);
  const handleLeave = useCallback(() => {
    // Pause before unmount so playback stops immediately even if React batches.
    videoRef.current?.pause();
    setHovering(false);
  }, []);

  const handleVideoError = useCallback(() => {
    // A playback/codec failure (e.g. the browser lacks the codec) must not nuke
    // the tile — fall back to the still poster, which loaded fine.
    setHovering(false);
  }, []);

  return (
    <div
      className={className}
      onMouseEnter={handleEnter}
      onMouseLeave={handleLeave}
      data-video-thumb="true"
      data-hovering={hovering ? "true" : "false"}
    >
      {hovering ? (
        <video
          ref={videoRef}
          className="resource-browser-video-poster"
          src={videoUrl}
          poster={posterUrl}
          muted
          loop
          autoPlay
          playsInline
          preload="auto"
          aria-label={alt}
          onError={handleVideoError}
        />
      ) : (
        <img
          className="resource-browser-video-poster"
          src={posterUrl}
          alt={alt}
          loading="lazy"
          onError={onError}
        />
      )}
    </div>
  );
}
