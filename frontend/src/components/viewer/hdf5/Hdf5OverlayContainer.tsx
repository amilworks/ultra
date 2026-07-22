import { createContext, type ReactNode, useContext } from "react";

const Hdf5OverlayContainerContext = createContext<HTMLElement | null>(null);

export function Hdf5OverlayContainerProvider({
  children,
  container,
}: {
  children: ReactNode;
  container: HTMLElement | null;
}) {
  return (
    <Hdf5OverlayContainerContext.Provider value={container}>
      {children}
    </Hdf5OverlayContainerContext.Provider>
  );
}

export const useHdf5OverlayContainer = (): HTMLElement | null =>
  useContext(Hdf5OverlayContainerContext);
