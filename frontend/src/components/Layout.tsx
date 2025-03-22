
import React from "react";

interface LayoutProps {
  children: React.ReactNode;
}

const Layout: React.FC<LayoutProps> = ({ children }) => {
  return (
    <div className="min-h-screen bg-gradient-to-br from-background to-secondary/40 flex flex-col">
      <header className="py-6 px-8 backdrop-blur-sm border-b border-border/20 sticky top-0 z-10">
        <div className="max-w-6xl mx-auto w-full flex items-center justify-between">
          <div className="flex items-center gap-2">
            <h1 className="text-xl font-medium tracking-tight">Podcast AI</h1>
            <span className="text-xs px-2 py-0.5 bg-secondary rounded-full text-muted-foreground">
              Beta
            </span>
          </div>
        </div>
      </header>
      <main className="flex-1 py-8 px-4 sm:px-8 max-w-7xl mx-auto w-full">
        {children}
      </main>
      <footer className="py-6 px-8 border-t border-border/20 text-sm text-muted-foreground">
        <div className="max-w-6xl mx-auto w-full text-center">
          <p>© {new Date().getFullYear()} Podcast AI Generator. All rights reserved.</p>
        </div>
      </footer>
    </div>
  );
};

export default Layout;
